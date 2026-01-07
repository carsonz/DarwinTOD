import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from .esb_data import ESBData, ESBStrategy, ESBMetadata, StrategyMetadata


class ESBUtils:
    """ESB工具类，提供数据加载、保存、验证等功能"""
    
    @staticmethod
    def load_esb_data(file_path: str) -> ESBData:
        """从JSON文件加载ESB数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ESB file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据结构
        if not ESBUtils.validate_esb_data(data):
            raise ValueError(f"Invalid ESB data structure in file: {file_path}")
        
        # 转换为ESBData对象
        strategies = []
        for strategy_data in data['strategies']:
            # 处理metadata字典到StrategyMetadata对象的转换
            metadata_dict = strategy_data['metadata']
            strategy = ESBStrategy(
                id=strategy_data['id'],
                domains=strategy_data['domains'],
                content=strategy_data['content'],
                agent_type=strategy_data['agent_type'],
                reason=strategy_data['reason'],
                metadata=StrategyMetadata(
                    helpful_count=metadata_dict.get('helpful_count', 0),
                    harmful_count=metadata_dict.get('harmful_count', 0),
                    used_count=metadata_dict.get('used_count', 0),
                    generation=metadata_dict.get('generation', 1),
                    parents=metadata_dict.get('parents', []),
                    alive=metadata_dict.get('alive', True)
                )
            )
            strategies.append(strategy)
        
        # 处理metadata字典到ESBMetadata对象的转换
        metadata_dict = data['metadata']
        esb_data = ESBData(
            strategies=strategies,
            metadata=ESBMetadata(
                total_strategies=metadata_dict.get('total_strategies', 0),
                alive_strategies=metadata_dict.get('alive_strategies', 0),
                strategies=metadata_dict.get('strategies', []),
                domains=metadata_dict.get('domains', [])
            )
        )
        
        return esb_data
    
    @staticmethod
    def save_esb_data(esb_data: ESBData, file_path: str) -> None:
        """将ESB数据保存到JSON文件"""
        # 转换为字典格式
        strategies_dict = []
        for strategy in esb_data.strategies:
            strategy_dict = {
                'id': strategy.id,
                'domains': strategy.domains,
                'content': strategy.content,
                'agent_type': strategy.agent_type,
                'reason': strategy.reason,
                'metadata': {
                    'helpful_count': strategy.metadata.helpful_count,
                    'harmful_count': strategy.metadata.harmful_count,
                    'used_count': strategy.metadata.used_count,
                    'generation': strategy.metadata.generation,
                    'parents': strategy.metadata.parents,
                    'alive': strategy.metadata.alive
                }
            }
            strategies_dict.append(strategy_dict)
        
        data = {
            'strategies': strategies_dict,
            'metadata': {
                'total_strategies': esb_data.metadata.total_strategies,
                'alive_strategies': esb_data.metadata.alive_strategies,
                'strategies': esb_data.metadata.strategies,
                'domains': esb_data.metadata.domains
            }
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def validate_esb_data(data: Dict[str, Any]) -> bool:
        """验证ESB数据结构的有效性"""
        # 检查顶层字段
        if 'strategies' not in data or 'metadata' not in data:
            return False
        
        # 检查strategies字段
        strategies = data['strategies']
        if not isinstance(strategies, list):
            return False
        
        # 检查每个策略
        for strategy in strategies:
            # 检查策略必填字段
            required_fields = ['id', 'domains', 'content', 'agent_type', 'reason', 'metadata']
            for field in required_fields:
                if field not in strategy:
                    return False
            
            # 检查字段类型
            if not isinstance(strategy['id'], str):
                return False
            if not isinstance(strategy['domains'], list):
                return False
            if not isinstance(strategy['content'], str):
                return False
            if not isinstance(strategy['agent_type'], str):
                return False
            if not isinstance(strategy['reason'], str):
                return False
            if not isinstance(strategy['metadata'], dict):
                return False
        
        # 检查metadata字段（如果是字典格式，允许部分字段缺失）
        metadata = data['metadata']
        if not isinstance(metadata, dict):
            return False
        
        # 检查metadata基本字段（允许缺失，因为可能是较老的格式）
        if 'strategies' not in metadata:
            return False
        
        return True
    
    @staticmethod
    def update_strategy_metadata(strategy: ESBStrategy) -> ESBStrategy:
        """更新单个策略的metadata，直接统计"""
        # 导入StrategyMetadata类
        from .esb_data import StrategyMetadata
        
        if strategy.metadata is None:
            strategy.metadata = StrategyMetadata()
        
        # 确保所有必填字段都有默认值（dataclass已经有默认值，但我们重新创建一个以确保）
        helpful_count = strategy.metadata.helpful_count
        harmful_count = strategy.metadata.harmful_count  
        used_count = strategy.metadata.used_count
        generation = strategy.metadata.generation
        parents = strategy.metadata.parents
        alive = strategy.metadata.alive
        
        # 创建一个更新的StrategyMetadata对象
        updated_metadata = StrategyMetadata(
            helpful_count=helpful_count,
            harmful_count=harmful_count,
            used_count=used_count,
            generation=generation,
            parents=parents,
            alive=alive
        )
        
        strategy.metadata = updated_metadata
        
        return strategy
    
    @staticmethod
    def update_esb_metadata(esb_data: ESBData) -> ESBData:
        """更新ESB整体metadata，直接统计"""
       
        # 统计策略总数
        total_strategies = len(esb_data.strategies)
        
        # 统计存活策略数
        alive_count = 0
        for strategy in esb_data.strategies:
            # 确保策略metadata有效
            ESBUtils.update_strategy_metadata(strategy)
            if strategy.metadata.alive:
                alive_count += 1
        
        # 统计strategies字段格式：[domains, agent_name, strategy_count, alive_count]
        strategies_info = []
        strategy_dict = {}  # 使用嵌套字典来聚合统计
        
        for strategy in esb_data.strategies:
            # 更新策略metadata
            ESBUtils.update_strategy_metadata(strategy)
            
            # 聚合每个domain-agent组合的统计信息
            agent_type = strategy.agent_type
            domains_tuple = tuple(sorted(strategy.domains))  # 将domains转换为元组作为key
            key = f"{domains_tuple}_{agent_type}"
            
            if key not in strategy_dict:
                strategy_dict[key] = {
                    'domains': list(domains_tuple),  # 转换为列表
                    'agent_name': agent_type,
                    'strategy_count': 0,
                    'alive_count': 0
                }
            
            strategy_dict[key]['strategy_count'] += 1
            if strategy.metadata.alive:
                strategy_dict[key]['alive_count'] += 1
        
        # 转换为列表格式
        strategies_info = list(strategy_dict.values())
        
        # 统计所有涉及的domains
        all_domains = set()
        for strategy in esb_data.strategies:
            all_domains.update(strategy.domains)
        
        # 创建新的ESBMetadata对象
        new_metadata = ESBMetadata(
            total_strategies=total_strategies,
            alive_strategies=alive_count,
            strategies=strategies_info,
            domains=list(all_domains)
        )
        
        esb_data.metadata = new_metadata
        
        return esb_data