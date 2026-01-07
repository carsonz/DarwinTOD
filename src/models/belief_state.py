"""
Belief state data model for IALM project.
Supports both MultiWOZ and SGD dataset formats.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
import json
import os


@dataclass
class BeliefState:
    """信念状态数据模型，支持MultiWOZ和SGD数据集格式"""
    dataset: str  # "multiwoz" 或 "sgd"
    domain: List[str]
    belief_state: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，验证数据集类型"""
        if self.dataset.lower() not in ["multiwoz", "sgd"]:
            raise ValueError(f"Unsupported dataset type: {self.dataset}. Must be 'multiwoz' or 'sgd'.")
    
    def update(self, updates: Dict[str, Any], confidence_updates: Optional[Dict[str, float]] = None) -> None:
        """
        更新信念状态。
        
        Args:
            updates: 槽位更新字典
            confidence_updates: 置信度更新字典
        """
        for slot, value in updates.items():
            self.belief_state[slot] = value
            
        if confidence_updates:
            for slot, conf in confidence_updates.items():
                self.confidence[slot] = conf
                
    def get_filled_slots(self) -> Dict[str, Any]:
        """获取已填充的槽位"""
        return {slot: value for slot, value in self.belief_state.items() if value is not None and value != ""}
        
    def get_slot_value(self, slot: str) -> Any:
        """获取槽位值"""
        return self.belief_state.get(slot)
        
    def get_slot_confidence(self, slot: str) -> float:
        """获取槽位置信度"""
        return self.confidence.get(slot, 0.0)
        
    def clear_slot(self, slot: str) -> None:
        """清除槽位值"""
        if slot in self.belief_state:
            del self.belief_state[slot]
        if slot in self.confidence:
            del self.confidence[slot]
            
    def clear_all_slots(self) -> None:
        """清除所有槽位"""
        self.belief_state.clear()
        self.confidence.clear()
        
    def is_slot_filled(self, slot: str) -> bool:
        """检查槽位是否已填充"""
        value = self.belief_state.get(slot)
        return value is not None and value != ""
        
    def get_required_slots(self, required_slots: List[str]) -> Dict[str, bool]:
        """
        检查必需槽位的填充状态。
        
        Args:
            required_slots: 必需槽位列表
            
        Returns:
            槽位填充状态字典
        """
        return {slot: self.is_slot_filled(slot) for slot in required_slots}
        
    def get_missing_slots(self, required_slots: List[str]) -> List[str]:
        """
        获取缺失的必需槽位。
        
        Args:
            required_slots: 必需槽位列表
            
        Returns:
            缺失的槽位列表
        """
        return [slot for slot in required_slots if not self.is_slot_filled(slot)]
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dataset": self.dataset,
            "domain": self.domain,
            "belief_state": self.belief_state,
            "confidence": self.confidence
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeliefState':
        """从字典创建实例"""
        return cls(
            dataset=data.get("dataset", "multiwoz"),
            domain=data.get("domain", ""),
            slots=data.get("belief_state", {}),
            confidence=data.get("confidence", {})
        )
        
    def to_json_string(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
        
    @classmethod
    def from_json_string(cls, json_str: str) -> 'BeliefState':
        """从JSON字符串创建实例"""
        data = json.loads(json_str)
        return cls.from_dict(data)
        
    def copy(self) -> 'BeliefState':
        """创建信念状态的副本"""
        return BeliefState(
            dataset=self.dataset,
            domain=self.domain,
            belief_state=self.belief_state.copy(),
            confidence=self.confidence.copy()
        )
        
    def merge(self, other: 'BeliefState') -> None:
        """
        合并另一个信念状态。
        
        Args:
            other: 另一个信念状态
        """
        if other.dataset != self.dataset:
            raise ValueError(f"Cannot merge belief states from different datasets: {self.dataset} vs {other.dataset}")
            
        # 检查domain列表是否有交集
        if not set(self.domain) & set(other.domain):
            raise ValueError(f"Cannot merge belief states from different domains: {self.domain} vs {other.domain}")
            
        self.belief_state.update(other.belief_state)
        self.confidence.update(other.confidence)
        # 合并domain列表，保持唯一性
        self.domain = list(set(self.domain + other.domain))
    
    @classmethod
    def create_default(cls, dataset: str, domain: List[str]) -> 'BeliefState':
        """创建数据集特定的默认信念状态"""
        if dataset not in ["multiwoz", "sgd"]:
            raise ValueError(f"Unsupported dataset: {dataset}")
            
        # 导入相应的state模块
        if dataset == "multiwoz":
            from convlab.util.multiwoz.state import default_state
            full_state = default_state()
        elif dataset == "sgd":
            from convlab.util.sgd.state import default_state
            full_state = default_state()
        
        # 获取完整的belief_state
        full_belief_state = full_state['belief_state']
        
        # 如果domain列表为空，返回空belief_state
        if not domain:
            return cls(dataset, [], {})
        
        belief_state = {}
        for d in domain:
            belief_state[d] = full_belief_state[d]

        return cls(dataset, domain, belief_state)


def parse_belief_state_from_response(previous_belief_state: Dict[str, Any], response_belief_state: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    从LLM响应中解析信念状态，更新slot值。
    
    Args:
        previous_belief_state: 前一轮的信念状态
        response_belief_state: LLM响应的信念状态（可以是JSON字符串或字典）
        
    Returns:
        更新后的信念状态，如果解析失败则返回previous_belief_state
    """
    try:
        # 如果response_belief_state是字符串，尝试解析为JSON
        if isinstance(response_belief_state, str):
            response_belief_state = json.loads(response_belief_state)
        
        # 如果不是字典类型，返回previous_belief_state
        if not isinstance(response_belief_state, dict):
            return previous_belief_state
        
        # 复制previous_belief_state作为基础
        updated_belief_state = previous_belief_state.copy()
        
        # 遍历response中的所有domain
        for domain, domain_data in response_belief_state.items():
            # 如果domain不存在，创建新的domain
            if domain not in updated_belief_state:
                updated_belief_state[domain] = {}
            
            # 确保domain_data是字典类型
            if isinstance(domain_data, dict):
                # 遍历domain中的所有slot
                for slot, value in domain_data.items():
                    # 更新slot值（包括新的slot）
                    updated_belief_state[domain][slot] = value
        
        return updated_belief_state
        
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        # 如果解析失败，返回previous_belief_state
        return previous_belief_state
