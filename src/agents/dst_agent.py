"""
Dialogue State Tracking (DST) Agent for IALM project.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from .base_agent import BaseAgent
from ..core.ssm import SharedStructuredMemory
from ..esb.esb_evolver import ESBEvolver as HSMEvolver
from ..models.belief_state import BeliefState, parse_belief_state_from_response

logger = logging.getLogger(__name__)


class DSTAgent(BaseAgent):
    """
    对话状态跟踪代理，负责解析用户话语，更新结构化对话状态。
    
    该代理接收用户话语和SSM对象作为输入，调用LLM API进行处理，
    生成更新后的信念状态，表示对用户目标、偏好和对话上下文的当前理解。
    """
    
    def __init__(self, config: Dict[str, Any], llm_client, ssm: SharedStructuredMemory, hsm_evolver: HSMEvolver = None, dataset_name: str = "multiwoz"):
        """
        初始化对话状态跟踪代理。
        
        Args:
            config: 代理配置字典
            llm_client: LLM客户端实例
            hsm: HSM工具类实例
            hsm_evolver: HSMEvolver实例，用于智能策略召回
            dataset_name: 数据集名称，支持"multiwoz"或"sgd"
        """
        super().__init__(config, llm_client, ssm, hsm_evolver)
        self.agent_type = "dst"
        self.dataset_name = dataset_name
    
    def process(self, domain: List[str], user_utterance: str, dialogue_history: List[str], pre_belief_state: Dict[str, Any], hsm: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理输入数据，生成更新后的信念状态。
        
        Args:
            domain: 领域列表
            user_utterance: 用户话语
            dialogue_history: 对话历史
            belief_state: 前一轮的信念状态
            prev_agent_result: 上一个agent的输出结果
            hsm: HSM策略
            
        Returns:
            包含更新后信念状态的字典
        """

        # 构造prompt
        prompt = self._construct_prompt(user_utterance, domain, dialogue_history, pre_belief_state, hsm)
        
        try:
            # 使用generate_with_chat_format接口
            messages = [
                {"role": "system", "content": "You are a dialogue state tracking agent for task-oriented conversations. Update structured belief states from user utterances. Only modify existing slot values, never add new slots or domains."},
                {"role": "user", "content": prompt}
            ]
            
            # 调用LLM生成响应
            response = self.llm_client.generate_with_chat_format(
                messages=messages
            )
            
            # 使用LLM客户端的clean_response方法清理响应
            cleaned_response = self.llm_client.clean_response(response)
            
            # 解析响应，提取信念状态、批判和理由
            parsed_response = json.loads(cleaned_response)
            updated_belief_state = parse_belief_state_from_response(pre_belief_state, parsed_response.get("belief_state", {}))
            criticism = parsed_response.get("criticism", "")
            reason = parsed_response.get("reason", "")
            
            return {
                "belief_state": updated_belief_state,
                "success": True,
                "criticism": criticism,
                "reason": reason
            }
            
        except Exception as e:
            self.logger.error(f"Error in DSTAgent.process: {str(e)}")
            # LLM调用失败，返回空信念状态
            return {
                "belief_state": belief_state,
                "success": False,
                "criticism": "",
                "reason": f"Error occurred: {str(e)}"
            }

    def _construct_prompt(self, user_utterance: str, domain: List[str], dialogue_history: List[str], 
                         previous_belief_state: Dict[str, Any], hsm: List[Dict[str, Any]]) -> str:
        """
        构造DST任务的prompt。
        
        Args:
            user_utterance: 用户话语
            domain: 对话领域
            dialogue_history: 对话历史
            previous_belief_state: 前一轮的信念状态
            hsm: HSM策略
            prev_agent_result: 上一个agent的输出结果
            
        Returns:
            构造好的prompt字符串
        """
        # 格式化对话历史
        formatted_history = ""
        if len(dialogue_history) > 0:
            formatted_history = "## Dialogue History:" + "\n".join(dialogue_history)
        
        # 格式化HSM策略
        formatted_hsm = ""
        if hsm and len(hsm) > 0:
            formatted_hsm = "## Relevant Strategies:\n"
            for strategy in hsm:
                formatted_hsm += strategy.get('content', '') + "\n"
        
        # 构造prompt
        prompt = f"""
## DIALOGUE STATE TRACKER
- Domain(s): {', '.join(domain)}
- User Utterance: {user_utterance}
{json.dumps(previous_belief_state, indent=2)}

{formatted_history}

{formatted_hsm}

## INSTRUCTIONS

### 1. Analyze User Utterance
- Extract slot value mentions from the user's current utterance
- Identify corrections, updates, or confirmations of existing values
- Analyze the previous User Output (user utterance). If there are any issues, provide criticism. If the output is good, leave criticism as empty string.

### 2. Update Belief State (CRITICAL RULES)
- Only modify existing slots: DO NOT create new slots or domains
- Corrections: If user corrects a slot (e.g., "actually I want X"), replace the old value
- Updates: If user provides new information for a slot, update it
- Persistence: If slot not mentioned, keep its current value unchanged
- Handling uncertainty: If utterance is ambiguous, prefer keeping current value unless clear update

### 3. Quality Check
- Verify all slot values are consistent with the utterance
- Ensure domain constraints are respected
- Check for contradictions between slots

## Output Format:
Output ONLY the JSON object. Do not include any additional text, explanations, or markdown formatting outside the JSON.
{{
  "criticism": "Your criticism of the User Output (if any)",
  "belief_state": {{"domain1": {{"slot1": "value1", "slot2": "value2"}}, "domain2": {{...}}}},
  "reason": "Explanation of belief state changes"
}}

"""
        return prompt
