"""
Natural Language Generation (NLG) Agent for IALM

This module implements the NLG agent which converts system actions into natural language responses.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from src.agents.base_agent import BaseAgent
from ..core.ssm import SharedStructuredMemory
from ..esb.esb_evolver import ESBEvolver as HSMEvolver
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class NLGAgent(BaseAgent):
    """
    Natural Language Generation (NLG) Agent
    
    This agent converts system actions into natural language responses.
    It takes system actions and SSM as input and generates natural language responses.
    """
    
    def __init__(self, config: Dict[str, Any], llm_client: LLMClient, ssm: SharedStructuredMemory, hsm_evolver: HSMEvolver = None):
        """
        Initialize the NLG Agent
        
        Args:
            config: Configuration dictionary
            llm_client: LLM client for generating responses
        """
        super().__init__(config, llm_client, ssm, hsm_evolver)
        self.agent_type = "nlg"       

    def process(self, domain: List[str], user_utterance: str, dialogue_history: List[str], belief_state: Dict[str, Any], system_action: str, db_results: Dict[str, Any], hsm: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the input data and generate a natural language response
        
        Args:
            domain: 对话领域列表
            user_utterance: 用户话语
            dialogue_history: 对话历史
            belief_state: 当前信念状态
            system_action: 系统动作
            db_results: 数据库查询结果
            prev_agent_result: 上一个agent的输出结果
            hsm: HSM策略
            
        Returns:
            Dictionary containing the generated response and updated SSM
        """        
        
        # Construct prompt
        prompt = self._construct_prompt(
            system_action=system_action,
            dialogue_history=dialogue_history,
            domain=domain,
            db_results=db_results,
            hsm=hsm,
            user_utterance=user_utterance
        )
        
        # Generate response using LLM with chat format
        try:
            messages = [
                {"role": "system", "content": "You are a Natural Language Generation agent for a task-oriented dialogue system. Convert system actions into natural language responses. Output your response in the specified JSON format."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.generate_with_chat_format(messages=messages)
            
            # 使用LLM客户端的clean_response方法清理响应
            cleaned_response = self.llm_client.clean_response(response)
            
            # Try to parse the cleaned response as JSON
            nlg_result = json.loads(cleaned_response)
            
            # Extract criticism and reason
            criticism = nlg_result.get("criticism", "")
            reason = nlg_result.get("reason", "")
            
            # Ensure it has the required field
            if "system_utterance" not in nlg_result:
                nlg_result = {"system_utterance": cleaned_response}
            
            # 添加criticism和reason到返回结果中
            nlg_result["criticism"] = criticism
            nlg_result["reason"] = reason
            
            return nlg_result
        except Exception as e:
            logger.error(f"Error parsing NLG response: {str(e)}")
            # If parsing fails, return as a dictionary with system_utterance
            return {
                "system_utterance": "I didn't understand, could you please say that again.",
                "criticism": "",
                "reason": f"Error occurred: {str(e)}"
            }

    def _construct_prompt(self, system_action: str, dialogue_history: List[str], 
                        domain: List[str], db_results: List[Dict[str, Any]], hsm: List[Dict[str, Any]], user_utterance: str) -> str:
        """
        Construct prompt for LLM
        
        Args:
            system_action: System action to convert
            dialogue_history: Previous dialogue turns
            domain: Current domain
            db_results: Database query results
            hsm_entries: Retrieved HSM strategies
            prev_agent_result: Previous agent's output result
            
        Returns:
            Constructed prompt string
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
                if isinstance(strategy, dict):
                    formatted_hsm += strategy.get('content', '') + "\n"
        
        # Format database results
        formatted_db_results = ""
        if len(db_results)>0:
            formatted_db_results = "- Database results:" + json.dumps(db_results)
        
        # Construct the full prompt
        prompt = f"""
## DIALOGUE STATE TRACKER
- Domain(s): {', '.join(domain)}
- User Utterance: {user_utterance}
- System action: {system_action}
{formatted_db_results}

{formatted_history}

{formatted_hsm}

## INSTRUCTIONS:
1. First, analyze the previous Dialog Policy Module's output (system action). If there are any issues, provide criticism. If the output is good, leave criticism as empty string.
2. Understand the intent behind the system action. Use the provided strategies to handle specific response patterns.
3. Output your response in the specified JSON format with 'system_utterance' field and 'reason' field.
4. Ensure the response is natural, helpful, and appropriate for the dialogue context. Keep the response concise but informative.

## Output Format:
Output ONLY the JSON object. Do not include any additional text, explanations, or markdown formatting outside the JSON.
{{
  "criticism": "Your criticism of the DP output (if any)",
  "system_utterance": "your natural system response",
  "reason": "Your reason for the NLG output"
}}
"""
        return prompt