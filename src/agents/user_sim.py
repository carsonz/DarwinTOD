"""
User Simulation Agent for IALM project.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from ..core.ssm import SharedStructuredMemory
from ..esb.esb_evolver import ESBEvolver as HSMEvolver

logger = logging.getLogger(__name__)


class UserSimAgent(BaseAgent):
    """
    用户模拟代理，用于模拟用户行为，根据输入生成符合用户特征的对话内容。
    
    该代理接收对话的goal（目标）和ssm（状态空间模型）作为输入，调用LLM API进行处理，
    生成符合用户模拟的角色设定、自然流畅且与对话目标相关的用户对话输出。
    """
    
    def __init__(self, config: Dict[str, Any], llm_client, ssm: SharedStructuredMemory, hsm_evolver: HSMEvolver = None):
        """
        初始化用户模拟代理。
        
        Args:
            config: 代理配置字典
            llm_client: LLM客户端实例
        """
        super().__init__(config, llm_client, ssm, hsm_evolver)
        self.agent_type = "user_sim"
        
    def process(self, domain: List[str], goal, belief_state, user_utterance, system_response, dialogue_history: List[str], prev_agent_result: Dict[str, Any] = None, hsm: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理输入数据，生成用户模拟响应。
        
        Args:
            domain: 对话领域列表
            goal: 用户目标
            belief_state: 当前信念状态
            user_utterance: 上一轮用户话语
            system_response: 系统响应
            dialogue_history: 对话历史
            prev_agent_result: 上一个agent的输出结果
            hsm: HSM策略
            
        Returns:
            包含用户响应的字典，格式为 {"user_utterance": "xxxxxx", "goal_achieved": true/false}
        """

        dialogue_history.append(f"user: {user_utterance}")
        dialogue_history.append(f"system: {system_response}")
        
        # 首先检查目标是否已经达成
        goal_achieved = self._check_goal_achieved(goal, belief_state)
        
        # 如果目标已达成，返回预定义的结束对话响应
        if goal_achieved:
            print("@@@@@@@@@@@@@@@@goal_achieved")
            return {
                "user_utterance": "Thank you for your help. That's all I need.",
                "goal_achieved": True,
                "criticism": "",
                "reason": "Goal achieved, ending dialogue"
            }
        
        # 构造prompt
        prompt = self._construct_prompt(domain, goal, belief_state, dialogue_history, hsm, prev_agent_result)
        
        try:
            # 使用generate_with_chat_format接口
            messages = [
                {"role": "system", "content": "You are a user simulation agent for a task-oriented dialogue system. Generate natural user responses based on dialogue context and user goal."},
                {"role": "user", "content": prompt}
            ]
            
            # 调用LLM生成响应
            response = self.llm_client.generate_with_chat_format(messages=messages)
            
            # 使用LLM客户端的clean_response方法清理响应
            cleaned_response = self.llm_client.clean_response(response)
            
            # 解析JSON响应
            try:
                parsed_response = json.loads(cleaned_response)
            except json.JSONDecodeError :
                logger.error(f"Failed to parse JSON response: {cleaned_response}")
                return {
                    "user_utterance": "didn't understand, could you please say that again.",
                    "goal_achieved": False,
                    "error": "Failed to parse LLM response",
                    "criticism": "",
                    "reason": "Failed to parse LLM response"
                }
            
            # 验证响应格式
            user_utterance = parsed_response.get("user_utterance", "")
            goal_achieved = parsed_response.get("goal_achieved", False)
            criticism = parsed_response.get("criticism", "")
            reason = parsed_response.get("reason", "")

            # 确保布尔值是正确的类型
            goal_achieved = self._ensure_boolean(goal_achieved)
                
            return {
                "user_utterance": user_utterance,
                "goal_achieved": goal_achieved,
                "criticism": criticism,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error in UserSimAgent.process: {str(e)}")
            # LLM调用失败，返回错误响应
            return {
                "user_utterance": "I didn't understand, could you please say that again.",
                "goal_achieved": False,
                "error": str(e),
                "criticism": "",
                "reason": f"Error in UserSimAgent.process: {str(e)}"
            }
    
    def _check_goal_achieved(self, goal: Dict[str, Any], belief_state: Dict[str, Any]) -> bool:
        """
        检查用户目标是否已在信念状态中完全实现
        
        Args:
            goal: 用户目标，包含inform和request部分
            belief_state: 当前信念状态
            
        Returns:
            如果目标已实现返回True，否则返回False
        """
        # 获取goal中的inform和request部分
        goal_inform = goal.get("inform", {})
        goal_request = goal.get("request", {})
        
        # 检查所有inform要求是否满足
        for domain, slots in goal_inform.items():
            if domain not in belief_state:
                return False
            for slot, expected_value in slots.items():
                # 检查槽位是否存在
                if slot not in belief_state[domain]:
                    return False
                
                current_value = belief_state[domain][slot]
                
                # 如果期望值为空或None，则认为只要槽位有值就满足条件
                if not expected_value:
                    continue
                
                # 如果期望值不为空，则检查是否匹配
                if "|" in expected_value:
                    # 处理包含"|"的期望值，表示"或"逻辑
                    expected_options = expected_value.split("|")
                    # 检查当前信念状态值是否匹配任何选项（忽略大小写）
                    if isinstance(current_value, str):
                        current_value_lower = current_value.lower().strip()
                        option_match = False
                        for option in expected_options:
                            option_lower = option.strip().lower()
                            if option_lower and current_value_lower == option_lower:
                                option_match = True
                                break
                        if not option_match:
                            return False
                    else:
                        # 非字符串类型，直接比较
                        if current_value not in expected_options:
                            return False
                else:
                    # 不包含"|"，进行直接比较（忽略大小写）
                    if isinstance(current_value, str) and isinstance(expected_value, str):
                        if current_value.lower().strip() != expected_value.lower().strip():
                            return False
                    else:
                        # 至少有一个不是字符串，进行直接比较
                        if current_value != expected_value:
                            return False
        
        # 检查所有request要求是否满足
        # for domain, slots in goal_request.items():
        #     if domain not in belief_state:
        #         return False
        #     for slot in slots:
        #         if slot not in belief_state[domain] or not belief_state[domain][slot]:
        #             return False
        
        return True
    
    def _construct_prompt(self, domain: List[str], goal, belief_state, dialogue_history: List[str], hsm: List[Dict[str, Any]], prev_agent_result: Dict[str, Any] = None) -> str:
        """
        构建用户模拟任务的prompt。
        
        Args:
            domain: 对话领域列表
            goal: 用户目标
            belief_state: 当前信念状态
            dialogue_history: 对话历史
            prev_agent_result: 上一个agent的输出结果
            
        Returns:
            构造好的prompt字符串
        """
        # 格式化对话历史
        formatted_history = "\n".join(dialogue_history)

        # 格式化HSM策略
        formatted_hsm = ""
        if hsm and len(hsm) > 0:
            formatted_hsm = "## Relevant Strategies:\n"
            for strategy in hsm:
                if isinstance(strategy, dict):
                    formatted_hsm += strategy.get('content', '') + "\n"
                    
        # 格式化前一个agent的输出
        formatted_prev_agent_output = ""
        if prev_agent_result:
            formatted_prev_agent_output = f"- Previous System Response:\n{json.dumps(prev_agent_result, indent=2)}\n\n"
        
        user_profile = "- User Profile: You are a smart and helpful user."
        # 构造prompt
        prompt = f"""
## USER SIMULATOR AGENT
- Domain: {", ".join(domain)}
- User Goal: {json.dumps(goal, indent=2)}
{user_profile}

## CURRENT DIALOGUE STATE
{formatted_prev_agent_output}
- Belief State: {json.dumps(belief_state, indent=2)}
- Dialogue History: {formatted_history}

{formatted_hsm}

## INSTRUCTIONS

### 1. Check Goal Completion
Compare belief state with user goal:
- Inform slots: All must match (OR conditions: match any "|" option)
- Only set `goal_achieved: true` when ALL requirements met

### 2. Generate Response
* If goal achieved: Natural closing (e.g., "Thanks, that's perfect!")

* If goal not achieved:
    - Identify the most urgent missing information
    - Generate natural follow-up question/statement
    - Avoid repeating same phrasing

### 3. Provide Reason
Explain in one sentences: response rationale

### 4. Criticism
Analyze the previous system response. If there are any issues, provide criticism. If the output is good, leave criticism as empty string.

## Output Format:
Output ONLY the JSON object. Do not include any additional text, explanations, or markdown formatting outside the JSON.
{{
  "criticism": "Constructive feedback on the previous system response (if any)",
  "user_utterance": "Natural user response that advances the dialogue",
  "goal_achieved": true/false,
  "reason": "Your reason for the user response"
}}
"""
        
        return prompt
    
    def _ensure_boolean(self, value: Any) -> bool:
        """
        确保输入值转换为布尔类型
        
        Args:
            value: 需要转换的值
            
        Returns:
            布尔值
        """
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ["true", "yes", "1", "on"]
        elif isinstance(value, int):
            return value != 0
        elif isinstance(value, float):
            return value != 0.0
        else:
            return bool(value)