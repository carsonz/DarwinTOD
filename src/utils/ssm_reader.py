#!/usr/bin/env python3
"""
SSM文件读取器，用于读取和解析IALM系统生成的ssm_XXXX.json文件
"""

import json
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class SSMDataReader:
    """
    SSM文件读取器，用于读取和解析IALM系统生成的ssm_XXXX.json文件
    """
    
    def __init__(self):
        """
        初始化SSMDataReader
        """
        pass
    
    def read_ssm_file(self, file_path: str) -> Dict[str, Any]:
        """
        读取并解析单个SSM文件
        
        Args:
            file_path: SSM文件路径
            
        Returns:
            解析后的SSM数据字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ssm_data = json.load(f)
            
            # 提取对话级信息
            dialogue_info = self._extract_dialogue_info(ssm_data)
            
            # 提取轮次级信息
            turn_info = self._extract_turn_info(ssm_data)
            
            # 提取Agent级信息
            agent_info = self._extract_agent_info(ssm_data)
            
            # 提取HSM策略信息
            hsm_info = self._extract_hsm_info(ssm_data)
            
            # 整合所有信息
            result = {
                "dialogue_info": dialogue_info,
                "turns": turn_info,
                "agent_info": agent_info,
                "hsm_info": hsm_info,
                "raw_data": ssm_data
            }
            
            logger.info(f"成功读取并解析SSM文件: {file_path}")
            return result
        except Exception as e:
            logger.error(f"读取SSM文件 {file_path} 出错: {str(e)}")
            raise
    
    def read_all_ssm_files(self, directory: str) -> List[Dict[str, Any]]:
        """
        读取目录中的所有SSM文件
        
        Args:
            directory: 包含SSM文件的目录路径
            
        Returns:
            解析后的SSM数据列表
        """
        ssm_files = []
        
        # 遍历目录，寻找所有ssm_*.json文件
        for filename in os.listdir(directory):
            if filename.startswith('ssm_') and filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                try:
                    ssm_data = self.read_ssm_file(file_path)
                    ssm_files.append(ssm_data)
                except Exception as e:
                    logger.error(f"处理SSM文件 {filename} 出错: {str(e)}")
        
        logger.info(f"成功读取 {len(ssm_files)} 个SSM文件")
        return ssm_files
    
    def _extract_dialogue_info(self, ssm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取对话级信息
        
        Args:
            ssm_data: 原始SSM数据
            
        Returns:
            对话级信息字典
        """
        dialogue_data = ssm_data.get('dialogues', {})
        
        dialogue_info = {
            "dialogue_id": dialogue_data.get('dialogue_id'),
            "domain": dialogue_data.get('domain', []),
            "goal": dialogue_data.get('goal', {}),
            "total_turns": len(dialogue_data.get('turns', []))
        }
        
        return dialogue_info
    
    def _extract_turn_info(self, ssm_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        提取轮次级信息
        
        Args:
            ssm_data: 原始SSM数据
            
        Returns:
            轮次级信息列表
        """
        turns = ssm_data.get('dialogues', {}).get('turns', [])
        turn_info_list = []
        
        for turn in turns:
            turn_info = {
                "turn_num": turn.get('turn_id'),
                "user": turn.get('user_utterance'),
                "system": turn.get('system_response'),
                "belief_state": turn.get('belief_state', {}),
                "system_action": turn.get('system_action'),
                "history": turn.get('history', []),
                "db_results": turn.get('db_results', {})
            }
            
            turn_info_list.append(turn_info)
        
        return turn_info_list
    
    def _extract_agent_info(self, ssm_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        提取Agent级信息
        
        Args:
            ssm_data: 原始SSM数据
            
        Returns:
            Agent级信息列表
        """
        turns = ssm_data.get('dialogues', {}).get('turns', [])
        agent_info_list = []
        
        for turn_num, turn in enumerate(turns):
            agent_data = turn.get('agent_data', {})
            
            for agent_type, data in agent_data.items():
                agent_info = {
                    "turn_num": turn_num + 1,
                    "agent_type": agent_type,
                    "prompt": data.get('prompt'),
                    "result": data.get('result'),
                    "clean_result": data.get('clean_result'),
                    "hsm_strategies": data.get('hsm', [])
                }
                agent_info_list.append(agent_info)
        
        return agent_info_list
    
    def _extract_hsm_info(self, ssm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取HSM策略信息
        
        Args:
            ssm_data: 原始SSM数据
            
        Returns:
            HSM策略信息字典
        """
        turns = ssm_data.get('dialogues', {}).get('turns', [])
        hsm_info = {
            "strategy_usage": {},
            "total_strategies": 0
        }
        
        for turn in turns:
            agent_data = turn.get('agent_data', {})
            for agent_type, data in agent_data.items():
                hsm_strategies = data.get('hsm', [])
                for strategy in hsm_strategies:
                    strategy_id = strategy.get('id')
                    if strategy_id:
                        hsm_info['strategy_usage'][strategy_id] = hsm_info['strategy_usage'].get(strategy_id, 0) + 1
                        hsm_info['total_strategies'] += 1
        
        return hsm_info
    
    def get_dialogue_ids(self, ssm_data_list: List[Dict[str, Any]]) -> List[str]:
        """
        获取对话ID列表
        
        Args:
            ssm_data_list: SSM数据列表
            
        Returns:
            对话ID列表
        """
        return [data['dialogue_info']['dialogue_id'] for data in ssm_data_list]
    
    def filter_by_domain(self, ssm_data_list: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """
        按领域过滤SSM数据
        
        Args:
            ssm_data_list: SSM数据列表
            domain: 领域名称
            
        Returns:
            过滤后的SSM数据列表
        """
        return [data for data in ssm_data_list if domain in data['dialogue_info']['domain']]
    
    def filter_by_turn_count(self, ssm_data_list: List[Dict[str, Any]], min_turns: int = None, max_turns: int = None) -> List[Dict[str, Any]]:
        """
        按轮次数量过滤SSM数据
        
        Args:
            ssm_data_list: SSM数据列表
            min_turns: 最小轮次数量
            max_turns: 最大轮次数量
            
        Returns:
            过滤后的SSM数据列表
        """
        filtered_data = ssm_data_list.copy()
        
        if min_turns is not None:
            filtered_data = [data for data in filtered_data if data['dialogue_info']['total_turns'] >= min_turns]
        
        if max_turns is not None:
            filtered_data = [data for data in filtered_data if data['dialogue_info']['total_turns'] <= max_turns]
        
        return filtered_data
