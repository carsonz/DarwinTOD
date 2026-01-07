"""
Base Agent class for IALM project.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from ..utils.llm_client import LLMClient
from src.core.ssm import SharedStructuredMemory
from src.esb.esb_evolver import ESBEvolver as HSMEvolver

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Agent基类，定义通用接口和属性。
    
    所有具体的Agent实现都必须继承自这个基类，并实现process方法。
    """
    
    def __init__(self, config: Dict[str, Any], llm_client: LLMClient, ssm: SharedStructuredMemory, hsm_evolver: HSMEvolver = None):
        """
        初始化Agent。
        
        Args:
            config: Agent配置字典
            llm_client: LLM客户端实例
        """
        self.config = config
        self.llm_client = llm_client
        self.agent_type = None
        self.hsm_evolver = hsm_evolver
        self.ssm = ssm
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 从配置中获取Agent特定参数
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 5000)
        
    @abstractmethod
    def process(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """
        处理输入数据，返回处理结果。
        
        Args:
            input_data: 输入数据，具体类型由子类定义
            context: 上下文信息，包含对话状态、历史等
            
        Returns:
            处理结果，具体类型由子类定义
        """
        pass
        