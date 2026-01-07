"""
Dialogue Policy (DP) Agent for IALM project.
"""

import json
import logging
import re
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from ..core.ssm import SharedStructuredMemory

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import LLMClient
from ..esb.esb_evolver import ESBEvolver as HSMEvolver

from data.multiwoz21.database import Database as MultiWOZDatabase
from data.sgd.database import Database as SGDDatabase


logger = logging.getLogger(__name__)


class DPAgent(BaseAgent):
    """
    Dialogue Policy (DP) Agent for task-oriented dialogue systems.
    
    The DP Agent is responsible for deciding the system action based on the current
    belief state and dialogue context. It determines whether a database query is needed,
    executes the query if necessary, and generates the appropriate system action.
    """
    
    def __init__(self, config: Dict[str, Any], llm_client: LLMClient, ssm: SharedStructuredMemory, hsm_evolver: HSMEvolver = None, dataset_type: str = "multiwoz"):
        """
        Initialize the DP Agent.
        
        Args:
            config: Agent configuration dictionary
            llm_client: LLM client instance
        """
        super().__init__(config, llm_client, ssm, hsm_evolver)
        self.agent_type = "dp"
        
        # Initialize dataset-specific components
        self.dataset_type = dataset_type
        
        # Initialize database based on dataset type
        self.db = None
        if self.dataset_type == "multiwoz" and MultiWOZDatabase:
            self.db = MultiWOZDatabase()
        elif self.dataset_type == "sgd" and SGDDatabase:
            self.db = SGDDatabase()

    def process(self, domain: List[str], user_utterance: str, dialogue_history: List[str], pre_belief_state: Dict[str, Any], belief_state: Dict[str, Any], hsm: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理输入数据，生成系统动作
        
        Args:
            domain: 对话领域
            user_utterance: 用户话语
            dialogue_history: 对话历史
            pre_belief_state: 前一轮的信念状态
            belief_state: 当前轮的信念状态
            prev_agent_result: 上一个agent的输出结果
            hsm: HSM策略
            
        Returns:
            包含系统动作的字典
        """
        
        # 构建提示并调用LLM
        prompt = self._construct_prompt(pre_belief_state, belief_state, user_utterance, dialogue_history, domain, hsm)
        query_params = ""
        
        try:
            # 调用LLM生成响应
            messages = [
                {"role": "system", "content": "You are a Dialogue Policy (DP) agent for a task-oriented dialogue system. Decide system actions based on belief state and dialogue context. Always query database for entity information, never use your own knowledge."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.generate_with_chat_format(
                messages=messages
            )
            
            # 使用LLM客户端的clean_response方法清理响应
            cleaned_response = self.llm_client.clean_response(response)

            # 解析JSON响应
            parsed_response = json.loads(cleaned_response)
            
            # 提取批判内容和理由
            criticism = parsed_response.get("criticism", "")
            reason = parsed_response.get("reason", "")
            
            # 检查是否需要查询数据库
            db_results = None
            if parsed_response.get("query_db", False):
                query_params = parsed_response.get("query", {})
                print(f"查询数据库：{query_params}")
                # 检查并补全state中缺失的domain
                query_domain = query_params.get("domain", "")
                if query_domain:
                    state = query_params.get("state", {})
                    # 如果state中缺少domain键，则补上
                    if query_domain not in state:
                        # 将现有的state作为该domain的值
                        if state:  # 如果state不为空
                            query_params["state"] = {query_domain: state}
                        else:  # 如果state为空，创建一个包含该domain的空字典
                            query_params["state"] = {query_domain: {}}
                        print(f"已补全缺失的domain '{query_domain}' 到state中")
                
                db_results = self._execute_database_query(query_params)
            
            # 返回结构化响应
            # 确保system_action是字符串类型
            system_action_raw = parsed_response.get("system_action", "")
            system_action = system_action_raw if isinstance(system_action_raw, str) else str(system_action_raw)
            
            return {
                "system_action": system_action,
                "db_query_needed": parsed_response.get("query_db", False),
                "query": parsed_response.get("query", {}),
                "db_results": db_results,
                "criticism": criticism,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error in DP Agent processing: {str(e)}")
            return {
                "system_action": "request(general->clarification)",
                "db_query_needed": False,
                "query": {},
                "db_results": None,
                "criticism": "",
                "reason": f"Error occurred: {query_params}"
            }
            
    def _construct_prompt(
        self, 
        pre_belief_state: Dict[str, Any], 
        belief_state: Dict[str, Any], 
        user_utterance: str, 
        dialogue_history: List[str], 
        domain: List[str], 
        hsm: Any
    ) -> str:
        """
        构建增强的提示模板，用于生成系统动作和数据库查询决策
        
        Args:
            belief_state: 当前信念状态
            user_utterance: 用户话语
            dialogue_history: 对话历史
            domain: 对话领域
            hsm_entries: 检索到的HSM策略
            prev_agent_result: 上一个agent的输出结果
            
        Returns:
            构建的提示字符串
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
        
        # 构建完整的提示
        prompt = f"""
## DIALOGUE POLICY AGENT        
- Domain(s): {', '.join(domain)}
- Lasted User Utterance: {user_utterance}
- Current Belief State: {json.dumps(belief_state, indent=2)}
- Previous Belief State:{json.dumps(pre_belief_state, indent=2)}

{formatted_history}

{formatted_hsm}

## INSTRUCTIONS:
1. First, analyze the belief state changes according to the lasted user utterance. If there are any issues, provide criticism. If the output is good, leave criticism as empty string.
2. Analyze the user's utterance and current belief state to determine the appropriate system action
3. CRITICAL: ALWAYS query the database for any information needed, NEVER use your own knowledge or common sense
4. Set "query_db" to true for ALL actions that require entity information, and specify the query parameters using the filled slots
5. Generate system action based ONLY on database query results, NEVER fabricate or assume any entity details, prices, addresses, or availability
6. Provide a reason for the DP output

## SYSTEM ACTION TYPES:
- inform(slot=value): Provide information to the user about a specific slot
- request(slot): Request more information from the user for a specific slot
- recommend(entity): Recommend a specific entity to the user
- select(entity): Select an entity from the database results
- nooffer(): Inform user that no matching results were found
- book(slot1=value1,slot2=value2): Make a booking with specified parameters
- nobook(): Inform user that booking cannot be completed
- offerbook(slot1=value1,slot2=value2): Offer booking options with specified parameters
- offerbooked(booking_details): Confirm successful booking with details

## Output Format:
Output ONLY the JSON object. Do not include any additional text, explanations, or markdown formatting outside the JSON.
{{
  "criticism": "Your criticism of the Belief State Changes (if any)",
  "system_action": "appropriate action based on the current context",
  "reason": "Your reason for the DP output",
  "query_db": true/false,
  "query": {{
    "domain": "the name of domain to query",
    "state": {{"the name of domain to query": {{"slot_name1": "value1", "slot_name2": "value2"}} }}
  }}
}}
"""

        return prompt
    
    def _execute_database_query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        执行数据库查询
        
        Args:
            domain: 对话领域
            query_params: 查询参数，应包含database.query()函数所需的所有参数
            
        Returns:
            查询结果列表
        """
        if not self.db:
            logger.warning("Database not initialized")
            return []
            
        try:                
            # 检查domain参数
            query_domain = query_params.get("domain", None)
            if not query_domain:
                logger.error("Domain not specified in query_params")
                return []
                
            # 获取state参数
            state = query_params.get("state", {})
            if not state:
                logger.warning("State not specified in query_params, using empty state")
                state = {}
                
            # 获取其他可选参数，设置默认值
            topk = query_params.get("topk", 5)
            ignore_open = query_params.get("ignore_open", False)
            soft_constraints = query_params.get("soft_constraints", [])
            fuzzy_match_ratio = query_params.get("fuzzy_match_ratio", 60)
            
            # 记录查询参数
            logger.debug(f"Executing database query with params: domain={query_domain}, state={state}, topk={topk}")
            
            # 执行查询
            results = self.db.query(
                domain=query_domain,
                state=state,
                topk=topk,
                ignore_open=ignore_open,
                soft_contraints=soft_constraints,  # 注意：数据库函数中使用的是soft_contraints而不是soft_constraints
                fuzzy_match_ratio=fuzzy_match_ratio
            )
            
            # 验证结果
            if not isinstance(results, list):
                logger.error(f"Invalid query result type: {type(results)}, expected list")
                return []
                
            logger.info(f"Database query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying database for domain {query_domain}: {str(e)}")
            return []
