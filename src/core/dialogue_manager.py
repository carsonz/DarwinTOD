"""
Dialogue Manager for IALM project.

This module implements the dialogue manager that orchestrates the conversation flow
between different agents (DST, DP, NLG, User Sim) and manages the dialogue state.
"""

import json
import logging
import os
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


from src.agents.dst_agent import DSTAgent
from src.agents.dp_agent import DPAgent
from src.agents.nlg_agent import NLGAgent
from src.agents.user_sim import UserSimAgent
from src.core.ssm import SharedStructuredMemory
from src.utils.llm_client import LLMClient
from src.utils.data_loader import DataLoader
from src.models.belief_state import BeliefState
from src.esb.esb_evolver import ESBEvolver

logger = logging.getLogger(__name__)


class DialogueState(Enum):
    """对话状态枚举"""
    INIT = "init"  # 初始状态
    ACTIVE = "active"  # 对话进行中
    SUCCESS = "success"  # 成功结束
    FAILURE = "failure"  # 失败结束
    COMPLETED = "completed"  # 对话完成


class DialogueManager:
    """
    对话管理器，负责协调各Agent完成对话流程
    
    该类负责:
    1. 初始化对话和各组件
    2. 管理对话状态转换
    3. 协调DST、DP、NLG和User Sim Agent的执行
    4. 与SSM交互，存储和检索对话数据
    5. 处理对话终止条件
    """
    
    def __init__(self, config: Dict[str, Any], llm_client: LLMClient, output_dir: str):
        """
        初始化对话管理器
        
        Args:
            config: 配置字典
            llm_client: LLM客户端实例
            output_dir: 输出目录
        """
        self.config = config
        self.llm_client = llm_client
        self.output_dir = output_dir

        # 初始化数据加载器
        dataset_config = config.get("data", {})
        self.dataset_type = dataset_config.get("default_dataset", "multiwoz")        
        self.data_loader = DataLoader(
            dataset_type=self.dataset_type,
            data_dir=dataset_config.get("data_dir", "data")
        )
        
        # 初始化SSM
        ssm_config = config.get("memory", {}).get("ssm", {})
        self.ssm = SharedStructuredMemory(ssm_config)
        
        # 初始化ESB
        self.esb_evolver = ESBEvolver()
        
        # 初始化各Agent
        agent_config = config.get("agents", {})
        
        self.dst_agent = DSTAgent(agent_config.get("dst", {}), llm_client, self.ssm, self.esb_evolver, self.dataset_type)
        self.dp_agent = DPAgent(agent_config.get("dp", {}), llm_client, self.ssm, self.esb_evolver, self.dataset_type)
        self.nlg_agent = NLGAgent(agent_config.get("nlg", {}), llm_client, self.ssm, self.esb_evolver)
        self.user_sim_agent = UserSimAgent(agent_config.get("user_sim", {}), llm_client, self.ssm, self.esb_evolver)
        
        # 对话状态
        self.current_dialogue_id = None
        self.current_state = DialogueState.INIT
        self.turn_count = 0
        self.max_turns = dataset_config.get("max_dialogue_turns", 40)
        self.goal = {}
        self.domain = []
        self.belief_state = None
        
        # 初始化全局对话计数器
        self.global_dialogue_count = 0
        
        # 存储每个agent在当前对话中使用的策略
        self.agent_strategies = {
            'dst': None,
            'dp': None,
            'nlg': None,
            'user_sim': None
        }
    
    def start_new_dialogue(self, dialogue_id: str, domains: List[str] ,goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        开始新对话
        
        Args:
            dialogue_id: 对话ID
            domains: 对话领域列表
            goal: 对话目标
            
        Returns:
            开始对话的结果，包含初始用户话语
        """
        self.current_dialogue_id = dialogue_id
        self.current_state = DialogueState.INIT
        self.turn_count = 0
        self.goal = goal
        self.domains = domains
        self.belief_state = BeliefState.create_default(self.dataset_type, domains)
        
        # 初始化agent策略字典
        self.agent_strategies = {
            'dst': None,
            'dp': None,
            'nlg': None,
            'user_sim': None
        }
        
        # 为每个agent加载或生成策略
        for agent_type in ['dst', 'dp', 'nlg', 'user_sim']:
            logger.info(f"Loading strategy for agent {agent_type} with domains {domains}...")
            
            # 构建查询信息
            query_info = {
                'agent_type': agent_type,
                'domains': domains,
                'top_k': 1,  # 只获取1个策略
                'temperature': 0.5
            }
            
            # 检索策略
            strategies = self.esb_evolver.recall_strategies(query_info)
            
            # 存储策略
            if strategies:
                # 直接使用返回的第一个策略，保持列表格式
                self.agent_strategies[agent_type] = strategies
                print(strategies)
                logger.info(f"Loaded strategy for {agent_type}: {strategies[0]['id']} with domains {strategies[0]['domains']}")
            else:
                logger.warning(f"Failed to load strategy for {agent_type} in domains {domains}")
        
        # 在SSM中创建新对话
        self.ssm.create_dialogue(dialogue_id, domains, goal)
        initial_user_utterance = self.data_loader.get_initial_user_utterance(dialogue_id)
        
        # 使用全局对话计数器，确保每个对话都有唯一的索引
        self.global_dialogue_count += 1
        self.dialogue_count = self.global_dialogue_count
        
        self.ssm.add_turn(dialogue_id, initial_user_utterance, self.belief_state.belief_state)
        
        # 更新状态为ACTIVE
        self.current_state = DialogueState.ACTIVE
        
        return {
            "user_utterance": initial_user_utterance,
            "state": self.current_state.value
        }
    
    def process_turn(self, dialogue_id: str) -> Dict[str, Any]:
        """
        处理一轮对话
        
        Args:
            dialogue_id: 对话ID
            
        Returns:
            处理结果，包含系统响应和对话状态
        """
        start_time = time.time()
        self.turn_count += 1
        
        # 初始化need_evovle列表
        need_evovle = []
        
        # 获取当前对话belief state
        dialogue_dict = self.ssm.get_dialogue(dialogue_id)
        
        # 获取最后一轮对话的用户话语
        last_turn_dict = dialogue_dict["turns"][-1]
        user_utterance = last_turn_dict["user_utterance"]

        dialogue_history = self.ssm.get_dialogue_history(dialogue_id)
        pre_belief_state =  self.ssm.get_last_belief_state(dialogue_id)
        
        # 1. DST处理 - 更新信念状态       
        dst_result = self.dst_agent.process(self.domains, user_utterance, dialogue_history, pre_belief_state, self.agent_strategies['dst'])
        belief_state = dst_result.get("belief_state", {})
        dst_reason = dst_result.get("reason", "")
        dst_criticism = dst_result.get("criticism", "")
        
        # 收集DST的批判内容
        if dst_criticism:
            need_evovle.append({
                "agent_type": "user_sim",
                "reason": dst_criticism
            })
        
        # 2. DP处理 - 生成系统动作（DP Agent内部决定是否需要查询数据库）        
        dp_result = self.dp_agent.process(self.domains, user_utterance, dialogue_history, pre_belief_state, belief_state, self.agent_strategies['dp'])

        # 确保system_action是字符串类型
        system_action_raw = dp_result.get("system_action", "")
        if isinstance(system_action_raw, str):
            system_action = system_action_raw
        else:
            # 如果不是字符串，尝试转换为字符串
            logger.warning(f"system_action is not a string: {type(system_action_raw)}, value: {system_action_raw}")
            system_action = str(system_action_raw)
            
        db_query = ""
        db_results = []  # 初始化为空列表
        if dp_result.get("db_query_needed", False):
            db_query = dp_result.get("query", "")
            db_results = dp_result.get("db_results", [])        
        
        dp_reason = dp_result.get("reason", "")
        dp_criticism = dp_result.get("criticism", "")
        
        # 收集DP的批判内容
        if dp_criticism:
            need_evovle.append({
                "agent_type": "dst",
                "reason": dp_criticism
            })
        
        # 3. NLG处理 - 生成系统响应
        nlg_result = self.nlg_agent.process(self.domains, user_utterance, dialogue_history, belief_state, system_action, db_results, self.agent_strategies['nlg'])
        system_utterance = nlg_result.get("system_utterance", "")
        nlg_reason = nlg_result.get("reason", "")
        nlg_criticism = nlg_result.get("criticism", "")
        
        # 收集NLG的批判内容
        if nlg_criticism:
            need_evovle.append({
                "agent_type": "dp",
                "reason": nlg_criticism
            })
        
        # 4. 更新当前轮次的信念状态、系统响应和动作
        self.ssm.update_turn(dialogue_id,
            belief_state,
            system_action,
            system_utterance,
            db_results,
            dst_reason,
            dp_reason,
            nlg_reason,
            need_evovle=need_evovle
        )
        
        # 5. 使用User Sim Agent生成下一轮用户话语      
        user_sim_result = self.user_sim_agent.process(self.domains, self.goal, belief_state, user_utterance, system_utterance, dialogue_history, self.agent_strategies['user_sim'])
        usersim_reason = user_sim_result.get("reason", "")

        print("\n")
        print(self.goal["inform"])
        print("-"*100)
        print(f"User: {user_utterance}")
        print(f"System: {system_utterance}")
        print(f"User: {user_sim_result.get('user_utterance')}")
        print("-"*100)
        print(f"System Action: {system_action}")
        if dp_result.get("db_query_needed", False):
            print(f"DB Query: {db_query}")
            print(f"DB Results: {db_results}")
        print(f"Belief State: {belief_state}")
        print(f" Reasons - DST: {dst_reason} \n Reasons - DP: {dp_reason} \n Reasons - NLG: {nlg_reason} \n Reasons - UserSim: {usersim_reason}")
        print(f" Criticisms - DST: {dst_result.get('criticism', '')} \n Criticisms - DP: {dp_result.get('criticism', '')} \n Criticisms - NLG: {nlg_result.get('criticism', '')} \n Criticisms - UserSim: {user_sim_result.get('criticism', '')}")
        print("="*100)

        # 处理UserSimAgent的结果
        user_utterance = user_sim_result.get("user_utterance", "")
        usersim_reason = user_sim_result.get("reason", "")
        usersim_criticism = user_sim_result.get("criticism", "")
        goal_achieved = user_sim_result.get("goal_achieved", False)
        
        # 收集UserSim的批判内容
        if usersim_criticism:
            need_evovle.append({
                "agent_type": "nlg",
                "reason": usersim_criticism
            })
        
        # 更新UserSim的reason到当前轮次
        turn = self.ssm.memory.turns[-1] if self.ssm.memory.turns else None
        if turn:
            turn.usersim_reason = usersim_reason
        
        # 6. 检查对话终止条件
        if goal_achieved:
            if self.user_sim_agent._check_goal_achieved(self.goal, belief_state):
                self.current_state = DialogueState.SUCCESS
            else:
                self.current_state = DialogueState.COMPLETED
        elif self.turn_count >= self.max_turns:
            self.current_state = DialogueState.FAILURE
        else:
            self.current_state =  DialogueState.ACTIVE
        
        # 7. 如果对话未完成，添加下一轮对话到SSM
        self.ssm.add_turn(dialogue_id, user_utterance, belief_state)
        if self.current_state != DialogueState.ACTIVE:
            # 对话已经结束，直接补充system action和system utterance
            self.ssm.end_turn(dialogue_id)
        
        # 8. 记录处理时间
        processing_time = time.time() - start_time
        
        logger.info(f"Processed turn {self.turn_count} for dialogue {dialogue_id}, state: {self.current_state.value}")

        # 9. 处理ESB演进
        if self.current_state != DialogueState.ACTIVE:
            try:
                # 从SSM获取完整的对话数据
                comprehensive_dialogue_data = self.ssm.get_comprehensive_dialogue_data(dialogue_id)
                
                # 添加对话结果信息
                comprehensive_dialogue_data['result'] = self.current_state.value
                comprehensive_dialogue_data['success'] = (self.current_state == DialogueState.SUCCESS)
                comprehensive_dialogue_data['dialog_index'] = self.global_dialogue_count
                comprehensive_dialogue_data['dialogue_id'] = dialogue_id
                comprehensive_dialogue_data['strategies'] = self.agent_strategies

                #self.esb_evolver.process_evolve(comprehensive_dialogue_data)                
            except Exception as e:
                logger.error(f"ESB evolution process failed: {str(e)}")
                # ESB处理失败不应该影响对话继续进行

        
        return {
            "dialogue_id": dialogue_id,
            "goal": self.goal,
            "belief_state": belief_state,
            "turn_count": self.turn_count,
            "dialogue_state": self.current_state.value,
            "dialogue_complete": self.current_state != DialogueState.ACTIVE,
            "processing_time": processing_time
        }
    
    def run_dialogue(self, dialogue_id: str, domain: List[str], goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行完整对话，从开始到结束
        
        Args:
            dialogue_id: 对话ID
            domain: 对话领域列表
            goal: 对话目标
            
        Returns:
            对话结果，包含完整对话历史和最终状态
        """
        # 开始新对话
        self.start_new_dialogue(dialogue_id, domain, goal)
        
        # 初始化结果记录
        result = {
            "dialogue_id": dialogue_id,
            "domains": domain,
            "goal": goal,
            "final_state": None,
            "total_turns": 0,
            "total_time": 0
        }
        
        start_time = time.time()
        
        # 循环处理对话轮次，直到对话完成
        while self.current_state == DialogueState.ACTIVE and self.turn_count < self.max_turns:
            turn_result = self.process_turn(dialogue_id)
            
            # 检查是否完成
            if turn_result["dialogue_complete"]:
                result["dialogue_id"] = turn_result["dialogue_id"]
                result["goal"] = turn_result["goal"]
                result["belief_state"] = turn_result["belief_state"]
                break
        
        # 记录最终结果
        result["final_state"] = self.current_state.value
        result["total_turns"] = self.turn_count
        result["total_time"] = time.time() - start_time
        result["dialogue_index"] = self.dialogue_count
        
        # 添加agent_strategies到SSM数据中
        if hasattr(self.ssm, 'memory') and self.ssm.memory:
            # 将agent_strategies添加到SSMData对象中
            self.ssm.memory.agent_strategies = {
                agent_type: strategy 
                for agent_type, strategy in self.agent_strategies.items() 
                if strategy is not None
            }
        
        # 保存SSM到文件
        file_path = os.path.join(self.output_dir, f"ssm_{self.dialogue_count:05d}_{dialogue_id}.json")
        self.ssm.save_to_file(file_path, self.current_state.value,result["total_time"])
    
        logger.info(f"Dialogue {dialogue_id} completed with state: {self.current_state.value} after {self.turn_count} turns")
        
        return result
    
    def load_dialogue_goals(self, dataset_name: str, num_goals: int = None, data_split: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[List[str]], List[str]]:
        """
        从指定数据集加载对话目标
        
        Args:
            dataset_name: 数据集名称 ("multiwoz" 或 "sgd")
            num_goals: 加载的目标数量，None表示加载全部
            data_split: 可选的数据分割类型 ("train", "validation", "test") 或 None 表示加载所有对话
            
        Returns:
            Tuple包含对话目标列表、领域列表和对话ID列表
        """
        # 如果传入的数据集名称与当前不同，更新数据集类型和相关组件
        if dataset_name != self.dataset_type:
            self.dataset_type = dataset_name
            dataset_config = self.config.get("data", {})
            
            # 更新数据加载器
            self.data_loader = DataLoader(
                dataset_type=dataset_name,
                data_dir=dataset_config.get("data_dir", "data")
            )
            
            # 更新信念状态创建时使用的数据集类型
            self.belief_state = BeliefState.create_default(self.dataset_type, [])
            
            # 更新所有Agent的数据集类型和相关组件
            agent_config = self.config.get("agents", {})
            self.dst_agent = DSTAgent(agent_config.get("dst", {}), self.llm_client, self.ssm, self.esb_evolver, self.dataset_type)
            self.dp_agent = DPAgent(agent_config.get("dp", {}), self.llm_client, self.ssm, self.esb_evolver, self.dataset_type)
            self.nlg_agent = NLGAgent(agent_config.get("nlg", {}), self.llm_client, self.ssm, self.esb_evolver)
            self.user_sim_agent = UserSimAgent(agent_config.get("user_sim", {}), self.llm_client, self.ssm, self.esb_evolver)
        
        dialogues = self.data_loader.get_random_dialogues(num_goals, data_split=data_split)
        
        # 提取对话目标
        goals = []
        domain_list = []
        ids = []
        
        for i in range(num_goals):
            dialogue = dialogues[i]

            # 提取主要领域（排除general）
            domains = [d for d in dialogue.get("domains", []) if d != "general"]
            
            goals.append(dialogue.get("goal", {}))
            domain_list.append(domains)
            ids.append(dialogue.get("dialogue_id", ""))
        
        return goals, domain_list, ids
    
