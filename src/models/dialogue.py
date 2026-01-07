"""
Dialogue data models for IALM project.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
from datetime import datetime


@dataclass
class Turn:
    """对话轮次数据模型"""
    turn_id: int
    user_utterance: str
    system_utterance: str = ""
    belief_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    db_query: Optional[Dict[str, Any]] = None
    db_results: Optional[List[Dict[str, Any]]] = None
    system_action: str = ""
    hsm_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "turn_id": self.turn_id,
            "user_utterance": self.user_utterance,
            "system_utterance": self.system_utterance,
            "belief_state": self.belief_state,
            "db_query": self.db_query,
            "db_results": self.db_results,
            "system_action": self.system_action,
            "hsm_used": self.hsm_used,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Turn':
        """从字典创建实例"""
        return cls(
            turn_id=data.get("turn_id", 0),
            user_utterance=data.get("user_utterance", ""),
            system_utterance=data.get("system_utterance", ""),
            belief_state=data.get("belief_state", {}),
            db_query=data.get("db_query"),
            db_results=data.get("db_results"),
            system_action=data.get("system_action", ""),
            hsm_used=data.get("hsm_used", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class Dialogue:
    """对话数据模型"""
    dialogue_id: str
    domain: str
    turns: List[Turn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: Turn) -> None:
        """添加对话轮次"""
        self.turns.append(turn)
        
    def get_last_turn(self) -> Optional[Turn]:
        """获取最后一轮对话"""
        if not self.turns:
            return None
        return self.turns[-1]
        
    def get_turn_by_id(self, turn_id: int) -> Optional[Turn]:
        """根据ID获取对话轮次"""
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None
        
    def get_dialogue_history(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """获取对话历史"""
        history = []
        turns = self.turns[-max_turns:] if max_turns else self.turns
        
        for turn in turns:
            if turn.user_utterance:
                history.append({"role": "user", "content": turn.user_utterance})
            if turn.system_utterance:
                history.append({"role": "system", "content": turn.system_utterance})
                
        return history
        
    def get_current_belief_state(self) -> Dict[str, Dict[str, Any]]:
        """获取当前信念状态"""
        if not self.turns:
            return {}
        return self.turns[-1].belief_state
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dialogue_id": self.dialogue_id,
            "domain": self.domain,
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dialogue':
        """从字典创建实例"""
        turns = [Turn.from_dict(turn_data) for turn_data in data.get("turns", [])]
        return cls(
            dialogue_id=data.get("dialogue_id", ""),
            domain=data.get("domain", ""),
            turns=turns,
            metadata=data.get("metadata", {})
        )
        
    def save_to_json(self, file_path: str) -> None:
        """保存到JSON文件"""
        data = self.to_dict()
        data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load_from_json(cls, file_path: str) -> 'Dialogue':
        """从JSON文件加载"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)