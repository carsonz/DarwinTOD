from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class StrategyMetadata:
    """ESB策略元数据格式"""
    helpful_count: int = 0           # 有用次数
    harmful_count: int = 0           # 有害次数  
    used_count: int = 0              # 使用次数
    generation: int = 0              # 策略代数
    parents: List[str] = field(default_factory=list)  # 父策略ID列表
    alive: bool = True               # 是否存活


@dataclass
class ESBMetadata:
    """ESB整体元数据格式"""
    total_strategies: int = 0        # 策略总数
    alive_strategies: int = 0        # 存活策略数
    strategies: List[Dict[str, Any]] = field(default_factory=list)  # 策略统计信息：[domain, agent_name, strategy_count, alive_count]
    domains: List[str] = field(default_factory=list)      # 包含的领域


@dataclass
class ESBStrategy:
    """ESB策略数据结构"""
    id: str                          # 策略唯一标识
    domains: List[str]               # 适用领域列表
    content: str                     # 策略内容
    agent_type: str                  # 适用Agent类型(dst/dp/nlg/user_sim)
    reason: str                      # 策略生成原因
    metadata: StrategyMetadata = field(default_factory=StrategyMetadata)  # 策略元数据


@dataclass
class ESBData:
    """ESB完整数据结构"""
    strategies: List[ESBStrategy]    # 策略列表
    metadata: ESBMetadata = field(default_factory=ESBMetadata)           # ESB元数据
