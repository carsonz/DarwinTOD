"""
Shared Structured Memory (SSM) for IALM project.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SSMNeedEvolve:
    """SSM需要演化的策略数据结构"""
    agent_type: str  # Agent类型
    reason: str  # 演化原因


@dataclass
class SSMTurn:
    """SSM对话轮次数据结构"""
    turn_id: int  # 轮次ID
    user_utterance: str  # 用户话语
    usersim_reason: str = ""  # 用户模拟器原因
    belief_state: Dict[str, Any] = field(default_factory=dict)  # 信念状态
    dst_reason: str = ""  # DST原因
    db_query: Optional[Dict[str, Any]] = None  # 数据库查询
    db_results: List[Dict[str, Any]] = field(default_factory=list)  # 数据库查询结果
    system_action: str = ""  # 系统动作
    dp_reason: str = ""  # DP原因
    system_response: str = ""  # 系统响应
    nlg_reason: str = ""  # NLG原因
    esb_used: List[str] = field(default_factory=list)  # 使用的ESB策略列表
    need_evovle: List[SSMNeedEvolve] = field(default_factory=list)  # 需要演化的策略列表
    history: str = ""  # 历史对话
    agent_data: Dict[str, Any] = field(default_factory=dict)  # 各Agent数据
    
    def __getitem__(self, key: str) -> Any:
        """
        支持字典式访问，保持与原有代码的兼容性
        
        Args:
            key: 要访问的键
            
        Returns:
            对应的值
        """
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        支持字典式赋值，保持与原有代码的兼容性
        
        Args:
            key: 要赋值的键
            value: 要赋值的值
        """
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """
        支持in运算符，保持与原有代码的兼容性
        
        Args:
            key: 要检查的键
            
        Returns:
            是否包含该键
        """
        return hasattr(self, key)


@dataclass
class SSMData:
    """SSM完整数据结构"""
    dialogue_id: str  # 对话ID
    domains: List[str]  # 领域列表
    goals: Dict[str, Any]  # 目标
    turns: List[SSMTurn]  # 轮次列表
    agent_strategies: Dict[str, Any] = field(default_factory=dict)  # 各agent使用的策略
    
    def __getitem__(self, key: str) -> Any:
        """
        支持字典式访问，保持与原有代码的兼容性
        
        Args:
            key: 要访问的键
            
        Returns:
            对应的值
        """
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        支持字典式赋值，保持与原有代码的兼容性
        
        Args:
            key: 要赋值的键
            value: 要赋值的值
        """
        setattr(self, key, value)


class SSMUtils:
    """SSM工具类，提供JSON与Python类的双向转换功能"""
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SSMData:
        """
        将字典转换为SSMData对象
        
        Args:
            data: 包含SSM数据的字典
            
        Returns:
            SSMData对象
        """
        turns = []
        for turn_data in data.get("turns", []):
            need_evolve = []
            for evolve_data in turn_data.get("need_evovle", []):
                need_evolve.append(SSMNeedEvolve(**evolve_data))
            
            turn_data["need_evovle"] = need_evolve
            turns.append(SSMTurn(**turn_data))
        
        return SSMData(
            dialogue_id=data["dialogue_id"],
            domains=data.get("domains", []),
            goals=data.get("goals", {}),
            turns=turns
        )
    
    @staticmethod
    def to_dict(ssm_data: SSMData) -> Dict[str, Any]:
        """
        将SSMData对象转换为字典
        
        Args:
            ssm_data: SSMData对象
            
        Returns:
            包含SSM数据的字典
        """
        data = {
            "dialogue_id": ssm_data.dialogue_id,
            "domains": ssm_data.domains,
            "goals": ssm_data.goals,
            "agent_strategies": ssm_data.agent_strategies,
            "turns": []
        }
        
        for turn in ssm_data.turns:
            turn_dict = {
                "turn_id": turn.turn_id,
                "user_utterance": turn.user_utterance,
                "usersim_reason": turn.usersim_reason,
                "belief_state": turn.belief_state,
                "dst_reason": turn.dst_reason,
                "system_action": turn.system_action,
                "dp_reason": turn.dp_reason,
                "system_response": turn.system_response,
                "nlg_reason": turn.nlg_reason,
                "need_evovle": [],
                "history": turn.history
            }
            
            # 处理可选字段
            if turn.db_query:
                turn_dict["db_query"] = turn.db_query
            
            if turn.db_results:
                turn_dict["db_results"] = turn.db_results
            
            # 处理need_evovle列表
            for evolve in turn.need_evovle:
                turn_dict["need_evovle"].append({
                    "agent_type": evolve.agent_type,
                    "reason": evolve.reason
                })
            
            data["turns"].append(turn_dict)
        
        return data
    
    @staticmethod
    def load_from_file(file_path: str) -> SSMData:
        """
        从文件加载SSM数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            SSMData对象
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 检查是否是完整的SSM数据格式
        if "dialogues" in data:
            # 兼容旧格式
            data = data["dialogues"]
        
        return SSMUtils.from_dict(data)
    
    @staticmethod
    def save_to_file(ssm_data: SSMData, file_path: str, final_state: str = "", total_time: float = 0.0) -> None:
        """
        将SSM数据保存到文件
        
        Args:
            ssm_data: SSMData对象
            file_path: 文件路径
            final_state: 最终状态
            total_time: 总时间
        """
        data = {
            "dialogues": SSMUtils.to_dict(ssm_data),
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "final_state": final_state,
                "total_turns": len(ssm_data.turns),
                "total_time": total_time
            }
        }
        
        # 创建目录如果不存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


class SharedStructuredMemory:
    """
    Shared Structured Memory (SSM) for storing and managing dialogue data.
    
    SSM provides a centralized storage for dialogues, belief states, and other
    dialogue-related information that can be accessed and updated by different agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SSM.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        # 初始化memory结构，使用新的数据类
        self.memory: Optional[SSMData] = None

        load_path = self.config.get("load_path")
        if load_path and os.path.exists(load_path):
            self.memory = SSMUtils.load_from_file(load_path)
            logger.info(f"Loaded SSM data from {load_path}")
    
    def __getitem__(self, key: str) -> Any:
        """
        支持字典式访问，保持与原有代码的兼容性
        
        Args:
            key: 要访问的键
            
        Returns:
            对应的值
        """
        if key == "turns" and self.memory:
            return self.memory.turns
        
        if self.memory:
            return getattr(self.memory, key)
        
        raise ValueError("No dialogue data available")
    
    def create_dialogue(self, dialogue_id: str, domain: List[str], goal: Dict[str, Any]):
        """
        Create a new dialogue.
        
        Args:
            dialogue_id: ID for the new dialogue
            domain: Domain of the dialogue
            goal: Dialogue goal
        """
        # 参数验证
        if not isinstance(dialogue_id, str) or not dialogue_id:
            raise ValueError("dialogue_id must be a non-empty string")
        
        if not isinstance(domain, list):
            raise ValueError("domain must be a list")
        
        if not isinstance(goal, dict):
            raise ValueError("goal must be a dictionary")
        
        self.memory = SSMData(
            dialogue_id=dialogue_id,
            domains=domain,
            goals=goal,
            turns=[]  # 初始化为空列表，后续添加对话轮次
        )
        logger.info(f"Created new dialogue: {dialogue_id} in domains: {domain}")
    
    def add_turn(self, dialogue_id: str, user_utterance: str, belief_state: Dict[str, Any]):
        """
        Add a new turn to a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            user_utterance: User utterance for the new turn
            belief_state: Belief state for the new turn
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        if not isinstance(user_utterance, str):
            raise ValueError("user_utterance must be a string")
        
        if not isinstance(belief_state, dict):
            raise ValueError("belief_state must be a dictionary")
        
        turn_id = len(self.memory.turns) + 1
        
        # 构建历史记录
        history_parts = []
        for prev_turn in self.memory.turns:
            if prev_turn.user_utterance:
                history_parts.append(f"user: {prev_turn.user_utterance}")
            if prev_turn.system_response:
                history_parts.append(f"system: {prev_turn.system_response}")
        history = "\n".join(history_parts)
        
        # 创建新的对话轮次
        turn = SSMTurn(
            turn_id=turn_id,
            user_utterance=user_utterance,
            belief_state=belief_state,
            history=history
        )
        
        # 添加到对话的turns列表中
        self.memory.turns.append(turn)
        
        logger.debug(f"Added turn {turn_id} to dialogue {dialogue_id}")
    
    def update_turn(self, dialogue_id: str, belief_state: Dict[str, Any], system_action: str, 
                   system_utterance: str, db_results: List[Dict[str, Any]],
                   dst_reason: str = "", dp_reason: str = "", nlg_reason: str = "",
                   usersim_reason: str = "", need_evovle: List[Dict[str, Any]] = None) -> None:
        """
        Update an existing turn in a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            belief_state: Updated belief state
            system_action: System action
            system_utterance: System response
            db_results: Database query results
            dst_reason: DST agent reason
            dp_reason: DP agent reason
            nlg_reason: NLG agent reason
            usersim_reason: UserSim agent reason
            need_evovle: List of strategies that need to be evolved
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        if not self.memory.turns:
            raise ValueError("No turns in dialogue yet")
        
        if not isinstance(belief_state, dict):
            raise ValueError("belief_state must be a dictionary")
        
        # 调试信息：检查system_action的类型和值
        if not isinstance(system_action, str):
            logger.error(f"system_action type error: expected str, got {type(system_action)}, value: {system_action}")
            # 尝试转换为字符串
            if system_action is not None:
                system_action = str(system_action)
            else:
                system_action = ""
        
        if not isinstance(system_utterance, str):
            # 调试信息：检查system_utterance的类型和值
            logger.error(f"system_utterance type error: expected str, got {type(system_utterance)}, value: {system_utterance}")
            # 尝试转换为字符串
            if system_utterance is not None:
                system_utterance = str(system_utterance)
            else:
                system_utterance = ""
        
        if not isinstance(db_results, list):
            raise ValueError("db_results must be a list")
        
        # 获取最后一轮
        turn = self.memory.turns[-1]
        
        # 更新轮次信息
        turn.system_response = system_utterance
        turn.belief_state = belief_state
        turn.system_action = system_action
        turn.db_results = db_results
        turn.dst_reason = dst_reason
        turn.dp_reason = dp_reason
        turn.nlg_reason = nlg_reason
        turn.usersim_reason = usersim_reason
        
        # 更新需要演化的策略列表
        if need_evovle:
            for evolve_item in need_evovle:
                turn.need_evovle.append(SSMNeedEvolve(**evolve_item))
        
        logger.debug(f"Updated turn {turn.turn_id}")

    def end_turn(self, dialogue_id: str) -> None:
        """
        End a dialogue turn with a goodbye message.
        
        Args:
            dialogue_id: ID of the dialogue
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        if not self.memory.turns:
            raise ValueError("No turns in dialogue yet")
        
        # 获取最后一轮
        turn = self.memory.turns[-1]
        
        # 设置结束信息
        turn.system_response = "GoodBye."
        turn.system_action = "say goodbye"
        
        logger.debug(f"End turn {turn.turn_id}")
    
    def get_dialogue_history(self, dialogue_id: str) -> List[str]:
        """
        Get the dialogue history for a specific dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            List of dialogue turns in chronological order (excluding the last turn)
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        history = []
        # 取前n-1个轮次，不包括最后一个
        for turn in self.memory.turns[:-1]:
            if turn.user_utterance:
                history.append(f"user: {turn.user_utterance}")
            if turn.system_response:
                history.append(f"system: {turn.system_response}")
                
        return history
    
    def get_dialogue(self, dialogue_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a dialogue by ID.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            Dialogue dictionary or None if not found
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        return SSMUtils.to_dict(self.memory)
    
    def get_current_belief_state(self, dialogue_id: str) -> Dict[str, Any]:
        """
        Get the current belief state for a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            Current belief state
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        if not self.memory.turns:
            raise ValueError("No turns in dialogue yet")
        
        # 返回最后一轮的信念状态
        return self.memory.turns[-1].belief_state

    def get_last_belief_state(self, dialogue_id: str) -> Dict[str, Any]:
        """
        Get the belief state from the previous turn.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            Belief state from the previous turn, or current turn if only one turn exists
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        if not self.memory.turns:
            raise ValueError("No turns in dialogue yet")
        
        if len(self.memory.turns) > 1:
            return self.memory.turns[-2].belief_state
        else:
            return self.memory.turns[-1].belief_state
    
    def save_to_file(self, file_path: str, final_state: str, total_time: float) -> None:
        """
        Save all dialogues to a JSON file.
        
        Args:
            file_path: Path to the output file
            final_state: Final state of the dialogue
            total_time: Total time taken for the dialogue
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue data to save")
        
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("file_path must be a non-empty string")
        
        if not isinstance(final_state, str):
            raise ValueError("final_state must be a string")
        
        if not isinstance(total_time, (int, float)):
            raise ValueError("total_time must be a number")
        
        SSMUtils.save_to_file(self.memory, file_path, final_state, total_time)
        logger.info(f"Saved SSM data to {file_path}")
    
    def get_dialogue_turns(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """
        Get all turns for a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            List of all turns in the dialogue
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        # 将SSMTurn对象转换为字典列表
        return [{
            "turn_id": turn.turn_id,
            "user_utterance": turn.user_utterance,
            "system_response": turn.system_response,
            "belief_state": turn.belief_state,
            "system_action": turn.system_action
        } for turn in self.memory.turns]
    
    def get_belief_states_sequence(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """
        Get belief states for all turns in a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            List of belief states for each turn
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        belief_states = []
        for turn in self.memory.turns:
            belief_states.append(turn.belief_state)
        
        return belief_states
    
    def get_system_actions_sequence(self, dialogue_id: str) -> List[str]:
        """
        Get system actions for all turns in a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            List of system actions for each turn
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        system_actions = []
        for turn in self.memory.turns:
            system_actions.append(turn.system_action)
        
        return system_actions
    
    def get_comprehensive_dialogue_data(self, dialogue_id: str) -> Dict[str, Any]:
        """
        Get comprehensive dialogue data for HSM processing.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            Comprehensive dialogue data containing history, belief states, system actions, etc.
        """
        # 参数验证
        if not self.memory:
            raise ValueError("No dialogue created yet")
        
        if dialogue_id != self.memory.dialogue_id:
            raise ValueError(f"Dialogue {dialogue_id} not found")
        
        dialogue = self.memory
        
        # 获取对话目标
        goal = dialogue.goals
        
        # 获取对话历史文本
        history_text = []
        for i, turn in enumerate(self.memory.turns):
            text = ""
            if turn.user_utterance:
                text = f"user: {turn.user_utterance}"
            if turn.system_response:
                text += f"\nsystem: {turn.system_response}"
            history_text.append({"turn_idx": i+1, "content": text})
        
        # 获取每轮DST信息（信念状态序列）
        belief_states = self.get_belief_states_sequence(dialogue_id)
        
        # 获取每轮系统动作
        system_actions = self.get_system_actions_sequence(dialogue_id)
        
        # 构建need_evolve字段
        need_evolve_dict = {}
        
        # 遍历所有轮次
        for turn_index, turn in enumerate(self.memory.turns):
            # 遍历当前轮次的need_evovle列表（注意字段名拼写）
            for evolve_item in turn.need_evovle:
                agent_type = evolve_item.agent_type
                criticism = evolve_item.reason

                # 初始化agent_type对应的列表
                if agent_type not in need_evolve_dict:
                    need_evolve_dict[agent_type] = []
                
                # 根据agent_type获取对应的agent_reason
                agent_reason = ""
                bs = {}
                sa = {}
                if agent_type == "dst":
                    agent_reason = turn.dst_reason
                    # 使用turn_index而不是turn_index-1来获取当前轮的信念状态
                    if turn_index < len(belief_states):
                        bs = belief_states[turn_index]
                elif agent_type == "dp":
                    agent_reason = turn.dp_reason
                    # 使用turn_index而不是turn_index-1来获取当前轮的系统动作
                    if turn_index < len(system_actions):
                        sa = system_actions[turn_index]
                elif agent_type == "nlg":
                    agent_reason = turn.nlg_reason
                elif agent_type == "user_sim":
                    agent_reason = turn.usersim_reason
            
                
                # 构建条目数据
                entry_data = {
                    "turn_idx": turn_index+1,
                    "criticism": criticism,
                    "agent_reason": agent_reason
                }
                
                # 根据agent_type添加额外的数据
                if agent_type == "dst":
                    entry_data["belief_state"] = bs
                elif agent_type == "dp":
                    entry_data["system_action"] = sa
                
                # 添加到need_evolve_dict
                need_evolve_dict[agent_type].append(entry_data)
        
        return {
            "goal": goal,
            "domain": dialogue.domains,
            "turns": history_text,
            "belief_states": belief_states,
            "system_actions": system_actions,
            "total_turns": len(self.memory.turns),
            "need_evolve": need_evolve_dict
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dialogues in SSM.
        
        Returns:
            Dictionary containing statistics
        """
        if not self.memory:
            return {
                "total_turns": 0,
                "domain_distribution": 0
            }
        
        total_turns = len(self.memory.turns)
        
        domain_counts = len(self.memory.domains)
        
        return {
            "total_turns": total_turns,
            "domain_distribution": domain_counts
        }