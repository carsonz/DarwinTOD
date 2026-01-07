#!/usr/bin/env python3
"""
IALM系统事后指标分析脚本

该脚本用于对指定目录下的ssm_XXXX.json实验结果文件进行事后指标分析，
计算各项实验指标并生成可视化报告。
"""

import os
import sys
import json
import argparse
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

class PostAnalysis:
    """
    事后指标分析类
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        初始化事后分析类
        
        Args:
            input_dir: 包含ssm文件的输入目录
            output_dir: 输出结果目录
        """
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            raise ValueError(f"输入目录 {input_dir} 不存在")
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ssm_files = []
        self.experiment_data = []
        self.metrics_results = {}
        
        # 确保输出目录存在
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.visualizations_dir, exist_ok=True)
            logger.info(f"输出目录已创建: {self.output_dir}")
        except Exception as e:
            logger.error(f"创建输出目录失败: {str(e)}")
            raise
    
    def find_ssm_files(self) -> List[str]:
        """
        查找指定目录下所有ssm_XXXX.json文件
        
        Returns:
            ssm文件路径列表
        """
        logger.info(f"在目录 {self.input_dir} 中查找ssm文件...")
        ssm_files = []
        
        # 使用正则表达式匹配ssm_XXXX.json格式的文件
        pattern = r"^ssm_\d+.*\.json$"
        
        for filename in os.listdir(self.input_dir):
            if re.match(pattern, filename):
                file_path = os.path.join(self.input_dir, filename)
                ssm_files.append(file_path)
        
        # 按文件名中的数字排序
        def get_experiment_number(file_path):
            """从文件名中提取实验序号"""
            basename = os.path.basename(file_path)
            match = re.match(r"^ssm_(\d+)_.*\.json$", basename)
            if match:
                return int(match.group(1))
            return 0
        
        ssm_files.sort(key=get_experiment_number)
        
        logger.info(f"找到 {len(ssm_files)} 个ssm文件")
        self.ssm_files = ssm_files
        return ssm_files
    
    def parse_ssm_files(self) -> List[Dict[str, Any]]:
        """
        解析ssm文件，提取关键信息
        
        Returns:
            解析后的实验数据列表
        """
        logger.info("开始解析ssm文件...")
        experiment_data = []
        
        for file_path in self.ssm_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 从文件名中提取实验序号
                basename = os.path.basename(file_path)
                match = re.match(r"^ssm_(\d+)_.*\.json$", basename)
                experiment_number = int(match.group(1)) if match else 0
                
                # 提取domains字段（数组类型）- FIXED: field name is 'domains' not 'domain'
                domains = data.get('dialogues', {}).get('domains', [])
                
                # 提取metadata中的final_state字段
                final_state = data.get('metadata', {}).get('final_state', 'unknown')
                
                # 添加到实验数据列表
                experiment_data.append({
                    'file_path': file_path,
                    'experiment_number': experiment_number,
                    'domains': domains,
                    'final_state': final_state,
                    'raw_data': data
                })
                
            except Exception as e:
                logger.error(f"解析文件 {file_path} 出错: {str(e)}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                continue
        
        logger.info(f"成功解析 {len(experiment_data)} 个ssm文件")
        logger.info(f"示例domains数据: {experiment_data[0]['domains'] if experiment_data else 'None'}")
        logger.info(f"示例final_state数据: {experiment_data[0]['final_state'] if experiment_data else 'None'}")
        self.experiment_data = experiment_data
        return experiment_data
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        调用_calculate_detailed_metrics函数计算指标
        
        Returns:
            指标计算结果
        """
        logger.info("开始计算详细指标...")
        
        # Handle case when no experiment data is available
        if not self.experiment_data:
            logger.warning("No experiment data available, returning empty metrics")
            return {}
        
        try:
            from src.utils.metrics import MetricsCalculator
            from src.utils.evaluation_report_generator import EvaluationReportGenerator
            from src.utils.ssm_reader import SSMDataReader
            
            # 1. 从SSM文件中读取数据
            ssm_reader = SSMDataReader()
            
            # 2. 创建一个简化的结果列表，模拟原始results参数
            # 这个列表将用于basic_metrics计算
            basic_results = []
            for data in self.experiment_data:
                basic_results.append({
                    "dialogue_id": data['raw_data'].get('dialogues', {}).get('dialogue_id', f"dialogue_{data['experiment_number']}"),
                    "final_state": data['final_state'],
                    "total_turns": len(data['raw_data'].get('dialogues', {}).get('turns', [])),
                    "total_time": data['raw_data'].get('metadata', {}).get('total_time', 0),
                    "goal": data['raw_data'].get('dialogues', {}).get('goal', {}),
                    "turns": data['raw_data'].get('dialogues', {}).get('turns', []),
                    "domain": data['domains']
                })
            
            # 3. 计算基础指标
            basic_metrics = self._calculate_basic_metrics(basic_results)
            
            # 4. 调用SSM数据读取器读取所有SSM文件
            ssm_data_list = []
            for file_path in self.ssm_files:
                try:
                    ssm_data = ssm_reader.read_ssm_file(file_path)
                    ssm_data_list.append(ssm_data)
                except Exception as e:
                    logger.error(f"读取SSM文件 {file_path} 出错: {str(e)}")
                    continue
            
            # 5. 检查是否存在final_evaluation_results.json文件，它可能包含预计算的BLEU分数
            # 先检查output_dir，再检查input_dir
            final_results_path = os.path.join(self.output_dir, "final_evaluation_results.json")
            if not os.path.exists(final_results_path):
                final_results_path = os.path.join(self.input_dir, "final_evaluation_results.json")
            
            # 也检查当前工作目录下的report文件夹
            if not os.path.exists(final_results_path):
                final_results_path = os.path.join(os.getcwd(), "report", "final_evaluation_results.json")
            
            # 先初始化metrics_calculator，以便在需要时使用
            metrics_calculator = MetricsCalculator()
            
            if os.path.exists(final_results_path):
                try:
                    with open(final_results_path, 'r', encoding='utf-8') as f:
                        final_results = json.load(f)
                    logger.info("成功读取final_evaluation_results.json文件")
                    
                    # 使用预计算的指标，但检查是否包含BLEU分数
                    all_metrics = final_results
                    bleu_score = all_metrics.get('bleu')
                    logger.info(f"从final_evaluation_results.json获取的BLEU分数: {bleu_score}")
                    
                    # 如果BLEU分数为0或不存在，使用metrics_calculator重新计算NLG指标
                    if bleu_score is None or bleu_score == 0:
                        logger.info("final_evaluation_results.json中BLEU分数为0或不存在，重新计算NLG指标")
                        # 5.1 将SSM数据转换为metrics_calculator所需的格式
                        processed_results = []
                        for ssm_data in ssm_data_list:
                            dialogue_info = ssm_data['dialogue_info']
                            turns = ssm_data['turns']
                            raw_data = ssm_data['raw_data']
                            
                            # 查找对应的原始结果
                            original_result = next((r for r in basic_results if r['dialogue_id'] == dialogue_info['dialogue_id']), None)
                            
                            # 从raw_data中提取goal信息
                            if raw_data and 'dialogues' in raw_data and 'goals' in raw_data['dialogues']:
                                goal = raw_data['dialogues']['goals']
                            else:
                                goal = dialogue_info['goal']
                            
                            if original_result:
                                # 使用原始结果的final_state和total_time等信息
                                converted_result = {
                                    "dialogue_id": dialogue_info['dialogue_id'],
                                    "final_state": original_result['final_state'],
                                    "total_turns": dialogue_info['total_turns'],
                                    "total_time": original_result['total_time'],
                                    "goal": goal,
                                    "turns": turns,
                                    "ssm_data": ssm_data,  # 添加ssm数据到结果中
                                    "domain": original_result['domain']
                                }
                                processed_results.append(converted_result)
                            else:
                                # 如果没有原始结果，创建一个新的
                                converted_result = {
                                    "dialogue_id": dialogue_info['dialogue_id'],
                                    "final_state": raw_data['metadata']['final_state'] if raw_data and 'metadata' in raw_data else "unknown",
                                    "total_turns": dialogue_info['total_turns'],
                                    "total_time": raw_data['metadata']['total_time'] if raw_data and 'metadata' in raw_data else 0,
                                    "goal": goal,
                                    "turns": turns,
                                    "ssm_data": ssm_data,  # 添加ssm数据到结果中
                                    "domain": dialogue_info['domain']
                                }
                                processed_results.append(converted_result)
                        
                        # 计算NLG指标
                        nlg_metrics = metrics_calculator._calculate_nlg_metrics(processed_results)
                        # 更新all_metrics中的NLG指标
                        all_metrics['rouge_l'] = nlg_metrics['rouge_l']
                        all_metrics['meteor'] = nlg_metrics['meteor']
                        all_metrics['bleu'] = nlg_metrics['bleu']
                        logger.info(f"重新计算的BLEU分数: {nlg_metrics['bleu']}")
                except Exception as e:
                    logger.error(f"读取final_evaluation_results.json失败: {str(e)}")
                    # 如果读取失败，继续计算指标
                    # 5.1 将SSM数据转换为metrics_calculator所需的格式
                    processed_results = []
                    for ssm_data in ssm_data_list:
                        dialogue_info = ssm_data['dialogue_info']
                        turns = ssm_data['turns']
                        raw_data = ssm_data['raw_data']
                        
                        # 查找对应的原始结果
                        original_result = next((r for r in basic_results if r['dialogue_id'] == dialogue_info['dialogue_id']), None)
                        
                        # 从raw_data中提取goal信息
                        if raw_data and 'dialogues' in raw_data and 'goals' in raw_data['dialogues']:
                            goal = raw_data['dialogues']['goals']
                        else:
                            goal = dialogue_info['goal']
                        
                        if original_result:
                            # 使用原始结果的final_state和total_time等信息
                            converted_result = {
                                "dialogue_id": dialogue_info['dialogue_id'],
                                "final_state": original_result['final_state'],
                                "total_turns": dialogue_info['total_turns'],
                                "total_time": original_result['total_time'],
                                "goal": goal,
                                "turns": turns,
                                "ssm_data": ssm_data,  # 添加ssm数据到结果中
                                "domain": original_result['domain']
                            }
                            processed_results.append(converted_result)
                        else:
                            # 如果没有原始结果，创建一个新的
                            converted_result = {
                                "dialogue_id": dialogue_info['dialogue_id'],
                                "final_state": raw_data['metadata']['final_state'] if raw_data and 'metadata' in raw_data else "unknown",
                                "total_turns": dialogue_info['total_turns'],
                                "total_time": raw_data['metadata']['total_time'] if raw_data and 'metadata' in raw_data else 0,
                                "goal": goal,
                                "turns": turns,
                                "ssm_data": ssm_data,  # 添加ssm数据到结果中
                                "domain": dialogue_info['domain']
                            }
                            processed_results.append(converted_result)
                    
                    # 5.2 计算详细评估指标
                    all_metrics = metrics_calculator.calculate_all_metrics(processed_results)
            else:
                # 如果final_evaluation_results.json不存在，继续计算指标
                processed_results = []
                for ssm_data in ssm_data_list:
                    dialogue_info = ssm_data['dialogue_info']
                    turns = ssm_data['turns']
                    raw_data = ssm_data['raw_data']
                    
                    # 查找对应的原始结果
                    original_result = next((r for r in basic_results if r['dialogue_id'] == dialogue_info['dialogue_id']), None)
                    
                    # 从raw_data中提取goal信息
                    if raw_data and 'dialogues' in raw_data and 'goals' in raw_data['dialogues']:
                        goal = raw_data['dialogues']['goals']
                    else:
                        goal = dialogue_info['goal']
                    
                    if original_result:
                        # 使用原始结果的final_state和total_time等信息
                        converted_result = {
                            "dialogue_id": dialogue_info['dialogue_id'],
                            "final_state": original_result['final_state'],
                            "total_turns": dialogue_info['total_turns'],
                            "total_time": original_result['total_time'],
                            "goal": goal,
                            "turns": turns,
                            "ssm_data": ssm_data,  # 添加ssm数据到结果中
                            "domain": original_result['domain']
                        }
                        processed_results.append(converted_result)
                    else:
                        # 如果没有原始结果，创建一个新的
                        converted_result = {
                            "dialogue_id": dialogue_info['dialogue_id'],
                            "final_state": raw_data['metadata']['final_state'] if raw_data and 'metadata' in raw_data else "unknown",
                            "total_turns": dialogue_info['total_turns'],
                            "total_time": raw_data['metadata']['total_time'] if raw_data and 'metadata' in raw_data else 0,
                            "goal": goal,
                            "turns": turns,
                            "ssm_data": ssm_data,  # 添加ssm数据到结果中
                            "domain": dialogue_info['domain']
                        }
                        processed_results.append(converted_result)
                
                # 计算详细评估指标
                all_metrics = metrics_calculator.calculate_all_metrics(processed_results)
            
            # 6. 合并基础指标和详细指标，确保基础指标优先
            final_metrics = {**all_metrics, **basic_metrics}
            
            # 7. 确保Combine指标被正确计算
            inform_rate = final_metrics.get("inform_rate", 0.0)
            success_rate = final_metrics.get("success_rate", 0.0)
            bleu = final_metrics.get("bleu", 0.0)
            final_metrics["combine"] = (inform_rate + success_rate) * 0.5 + bleu
            logger.info(f"最终计算的Combine分数: {final_metrics['combine']}")
            
            # 8. 生成评估报告
            logger.info("生成评估报告...")
            report_generator = EvaluationReportGenerator()
            report_dir = report_generator.generate_comprehensive_report(
                processed_results if 'processed_results' in locals() else [], final_metrics, self.output_dir
            )
            
            # 9. 保存详细指标到单独文件
            metrics_file = os.path.join(self.output_dir, "detailed_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(final_metrics, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"评估完成，报告已保存到: {report_dir}")
            
            self.metrics_results = final_metrics
            return final_metrics
            
        except ImportError as e:
            logger.warning(f"评估模块导入失败，仅使用基础指标: {e}")
            # 仅返回基础指标
            basic_results = []
            for data in self.experiment_data:
                basic_results.append({
                    "dialogue_id": data['raw_data'].get('dialogues', {}).get('dialogue_id', f"dialogue_{data['experiment_number']}"),
                    "final_state": data['final_state'],
                    "total_turns": len(data['raw_data'].get('dialogues', {}).get('turns', [])),
                    "total_time": data['raw_data'].get('metadata', {}).get('total_time', 0),
                    "goal": data['raw_data'].get('dialogues', {}).get('goal', {}),
                    "turns": data['raw_data'].get('dialogues', {}).get('turns', []),
                    "domain": data['domains']
                })
            basic_metrics = self._calculate_basic_metrics(basic_results)
            return basic_metrics
        except Exception as e:
            import traceback
            logger.error(f"计算详细评估指标出错: {str(e)}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 仅返回基础指标
            basic_results = []
            for data in self.experiment_data:
                basic_results.append({
                    "dialogue_id": data['raw_data'].get('dialogues', {}).get('dialogue_id', f"dialogue_{data['experiment_number']}"),
                    "final_state": data['final_state'],
                    "total_turns": len(data['raw_data'].get('dialogues', {}).get('turns', [])),
                    "total_time": data['raw_data'].get('metadata', {}).get('total_time', 0),
                    "goal": data['raw_data'].get('dialogues', {}).get('goal', {}),
                    "turns": data['raw_data'].get('dialogues', {}).get('turns', []),
                    "domain": data['domains']
                })
            basic_metrics = self._calculate_basic_metrics(basic_results)
            return basic_metrics
    
    def _calculate_basic_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算基础指标
        
        Args:
            results: 对话结果列表
            
        Returns:
            基础指标字典
        """
        logger.info("开始计算基础指标...")
        
        if not results:
            return {}
        
        total_dialogues = len(results)
        success_count = 0
        complete_count = 0
        failure_count = 0
        success_total_turns = 0
        complete_total_turns = 0
        success_complete_total_turns = 0
        success_complete_count = 0
        total_turns = 0
        
        for result in results:
            final_state = result.get("final_state", "failed")
            total_turns += result.get("total_turns", 0)
            
            if final_state in ["success"]:
                success_count += 1
                success_total_turns += result.get("total_turns", 0)
                success_complete_total_turns += result.get("total_turns", 0)
                success_complete_count += 1
            elif final_state in ["completed"]:
                complete_count += 1
                complete_total_turns += result.get("total_turns", 0)
                success_complete_total_turns += result.get("total_turns", 0)
                success_complete_count += 1
            else:
                failure_count += 1
        
        basic_metrics = {
            "total_dialogues": total_dialogues,
            "success_count": success_count,
            "complete_count": complete_count,
            "failure_count": failure_count,
            "success_rate": success_count / total_dialogues if total_dialogues > 0 else 0,
            "complete_rate": complete_count / total_dialogues if total_dialogues > 0 else 0,
            "failure_rate": failure_count / total_dialogues if total_dialogues > 0 else 0,
            "average_turns": total_turns / total_dialogues if total_dialogues > 0 else 0,
            "success_avg_turns": success_total_turns / success_count if success_count > 0 else 0,
            "complete_avg_turns": complete_total_turns / complete_count if complete_count > 0 else 0,
            "success_complete_avg_turns": success_complete_total_turns / success_complete_count if success_complete_count > 0 else 0
        }
        
        logger.info(f"基础指标计算完成: {basic_metrics}")
        return basic_metrics
    
    def generate_visualizations(self) -> None:
        """
        生成数据可视化图表
        """
        logger.info("开始生成可视化图表...")
        
        try:
            # Handle case when no experiment data is available
            if not self.experiment_data:
                logger.warning("No experiment data available, skipping visualizations")
                return
            
            # 1. 将实验数据转换为DataFrame，方便后续处理
            df = pd.DataFrame(self.experiment_data)
            
            # 2. 生成表格1和折线图1：按实验序号每20个一组
            self._generate_table_and_chart_1(df)
            
            # 3. 生成表格2和折线图2：按domains分组，每10个相同domains的实验为一组
            self._generate_table_and_chart_2(df)
            
            logger.info("可视化图表生成完成")
        except Exception as e:
            logger.error(f"生成可视化图表失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def _generate_table_and_chart_1(self, df: pd.DataFrame) -> None:
        """
        生成表格1和折线图1：按实验序号每20个一组
        
        Args:
            df: 实验数据DataFrame
        """
        logger.info("生成表格1和折线图1...")
        
        # Handle case when DataFrame is empty
        if df.empty:
            logger.warning("Empty DataFrame, skipping table1 and chart1 generation")
            return
        
        # 2.1 按实验序号排序
        df_sorted = df.sort_values('experiment_number')
        
        # 2.2 每20个实验为一组，添加组编号
        df_sorted['group_id'] = (df_sorted.index // 20) + 1
        
        # 2.3 统计每组中三种状态的数量和平均对话轮次
        def count_states(group):
            """统计一组中三种状态的数量和平均对话轮次"""
            counts = group['final_state'].value_counts().to_dict()
            # 计算平均对话轮次，需要从raw_data中提取total_turns
            total_turns = 0
            for _, row in group.iterrows():
                raw_data = row['raw_data']
                # 从metadata中获取total_turns，如果没有则从turns列表长度计算
                turns = raw_data.get('dialogues', {}).get('turns', [])
                total_turns += len(turns)
            avg_turns = total_turns / len(group) if len(group) > 0 else 0
            return pd.Series({
                'success_count': counts.get('success', 0),
                'completed_count': counts.get('completed', 0),
                'failed_count': counts.get('failed', 0),
                'total': len(group),
                'average_turns': avg_turns
            })
        
        group_stats = df_sorted.groupby('group_id').apply(count_states).reset_index()
        
        # 2.4 计算比例
        group_stats['success_ratio'] = group_stats['success_count'] / group_stats['total']
        group_stats['completed_ratio'] = group_stats['completed_count'] / group_stats['total']
        group_stats['failed_ratio'] = group_stats['failed_count'] / group_stats['total']
        
        # 2.5 生成表格1
        table1 = group_stats[[
            'group_id', 'total', 'success_count', 'completed_count', 'failed_count',
            'average_turns', 'success_ratio', 'completed_ratio', 'failed_ratio'
        ]].copy()
        
        # 格式化表格
        table1['success_ratio'] = table1['success_ratio'].map('{:.2%}'.format)
        table1['completed_ratio'] = table1['completed_ratio'].map('{:.2%}'.format)
        table1['failed_ratio'] = table1['failed_ratio'].map('{:.2%}'.format)
        
        # 保存表格为CSV和markdown
        table1_path_csv = os.path.join(self.output_dir, 'table1.csv')
        table1_path_md = os.path.join(self.output_dir, 'table1.md')
        table1.to_csv(table1_path_csv, index=False)
        
        with open(table1_path_md, 'w', encoding='utf-8') as f:
            f.write('# Table 1: Grouped by Experiment Number (20 experiments per group)\n\n')
            f.write(table1.to_markdown(index=False))
            f.write('\n')
        
        # 2.6 生成折线图1
        plt.figure(figsize=(12, 6))
        
        # 左侧y轴：状态比例
        ax1 = plt.gca()
        ax1.plot(group_stats['group_id'], group_stats['success_ratio'], 
                 marker='o', linewidth=2, label='Success Ratio', color='#1f77b4')
        ax1.plot(group_stats['group_id'], group_stats['completed_ratio'], 
                 marker='s', linewidth=2, label='Completed Ratio', color='#ff7f0e')
        ax1.plot(group_stats['group_id'], group_stats['failed_ratio'], 
                 marker='^', linewidth=2, label='Failed Ratio', color='#2ca02c')
        
        ax1.set_title('Status Ratio and Average Turns Trend by Experiment Number (20 experiments per group)', fontsize=16)
        ax1.set_xlabel('Group ID', fontsize=12)
        ax1.set_ylabel('Status Ratio', fontsize=12, color='k')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.legend(loc='upper left')
        
        # 右侧y轴：平均对话轮次
        ax2 = ax1.twinx()
        ax2.plot(group_stats['group_id'], group_stats['average_turns'], 
                 marker='D', linewidth=2, label='Average Turns', color='#d62728')
        ax2.set_ylabel('Average Dialogue Turns', fontsize=12, color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        ax2.grid(False)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        chart1_path = os.path.join(self.visualizations_dir, 'chart1.png')
        plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"表格1和折线图1已保存到 {self.output_dir} 和 {self.visualizations_dir}")
    
    def _generate_table_and_chart_2(self, df: pd.DataFrame) -> None:
        """
        生成表格2和折线图2：按domains分组，每10个相同domains的实验为一组
        
        Args:
            df: 实验数据DataFrame
        """
        logger.info("生成表格2和折线图2...")
        
        # Handle case when DataFrame is empty
        if df.empty:
            logger.warning("Empty DataFrame, skipping table2 and chart2 generation")
            return
        
        # 3.1 按实验序号排序
        df_sorted = df.sort_values('experiment_number')
        
        # 3.2 将domains转换为可哈希的元组，用于分组
        df_sorted['domains_tuple'] = df_sorted['domains'].apply(tuple)
        
        # 3.3 为每个domains组合分配连续编号，用于识别相同domains的实验
        df_sorted['domains_id'] = df_sorted['domains_tuple'].rank(method='dense').astype(int)
        
        # 3.4 按domains_id分组，为每个domains组合内部的实验添加序号
        df_sorted['domain_group_seq'] = df_sorted.groupby('domains_id').cumcount() + 1
        
        # 3.5 每10个相同domains的实验为一组，添加组编号
        df_sorted['group_id'] = (df_sorted['domain_group_seq'] - 1) // 10 + 1
        
        # 3.6 为每个domains组合和组id创建唯一的组合键
        df_sorted['unique_group_key'] = df_sorted['domains_id'].astype(str) + '_' + df_sorted['group_id'].astype(str)
        
        # 3.7 统计每个唯一组合中三种状态的数量和平均对话轮次
        def count_states_2(group):
            """统计一组中三种状态的数量和平均对话轮次"""
            counts = group['final_state'].value_counts().to_dict()
            # 计算平均对话轮次，需要从raw_data中提取total_turns
            total_turns = 0
            for _, row in group.iterrows():
                raw_data = row['raw_data']
                # 从metadata中获取total_turns，如果没有则从turns列表长度计算
                turns = raw_data.get('dialogues', {}).get('turns', [])
                total_turns += len(turns)
            avg_turns = total_turns / len(group) if len(group) > 0 else 0
            return pd.Series({
                'success_count': counts.get('success', 0),
                'completed_count': counts.get('completed', 0),
                'failed_count': counts.get('failed', 0),
                'total': len(group),
                'average_turns': avg_turns,
                'domains': group['domains'].iloc[0]  # 取组内第一个domains作为代表
            })
        
        group_stats = df_sorted.groupby('unique_group_key').apply(count_states_2).reset_index()
        
        # 3.8 计算比例
        group_stats['success_ratio'] = group_stats['success_count'] / group_stats['total']
        group_stats['completed_ratio'] = group_stats['completed_count'] / group_stats['total']
        group_stats['failed_ratio'] = group_stats['failed_count'] / group_stats['total']
        
        # 3.9 为了可视化，我们需要一个连续的组编号
        group_stats['visual_group_id'] = range(1, len(group_stats) + 1)
        
        # 3.10 生成表格2
        table2 = group_stats[[
            'visual_group_id', 'domains', 'total', 'success_count', 'completed_count', 'failed_count',
            'average_turns', 'success_ratio', 'completed_ratio', 'failed_ratio'
        ]].copy()
        
        # 格式化表格
        table2['success_ratio'] = table2['success_ratio'].map('{:.2%}'.format)
        table2['completed_ratio'] = table2['completed_ratio'].map('{:.2%}'.format)
        table2['failed_ratio'] = table2['failed_ratio'].map('{:.2%}'.format)
        
        # 保存表格为CSV和markdown
        table2_path_csv = os.path.join(self.output_dir, 'table2.csv')
        table2_path_md = os.path.join(self.output_dir, 'table2.md')
        table2.to_csv(table2_path_csv, index=False)
        
        with open(table2_path_md, 'w', encoding='utf-8') as f:
            f.write('# Table 2: Grouped by Domains (10 experiments per group with same domains)\n\n')
            f.write(table2.to_markdown(index=False))
            f.write('\n')
        
        # 3.11 生成折线图2
        plt.figure(figsize=(12, 6))
        
        # 左侧y轴：状态比例
        ax1 = plt.gca()
        ax1.plot(group_stats['visual_group_id'], group_stats['success_ratio'], 
                 marker='o', linewidth=2, label='Success Ratio', color='#1f77b4')
        ax1.plot(group_stats['visual_group_id'], group_stats['completed_ratio'], 
                 marker='s', linewidth=2, label='Completed Ratio', color='#ff7f0e')
        ax1.plot(group_stats['visual_group_id'], group_stats['failed_ratio'], 
                 marker='^', linewidth=2, label='Failed Ratio', color='#2ca02c')
        
        ax1.set_title('Status Ratio and Average Turns Trend by Domains (10 experiments per group with same domains)', fontsize=16)
        ax1.set_xlabel('Group ID', fontsize=12)
        ax1.set_ylabel('Status Ratio', fontsize=12, color='k')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.legend(loc='upper left')
        
        # 右侧y轴：平均对话轮次
        ax2 = ax1.twinx()
        ax2.plot(group_stats['visual_group_id'], group_stats['average_turns'], 
                 marker='D', linewidth=2, label='Average Turns', color='#d62728')
        ax2.set_ylabel('Average Dialogue Turns', fontsize=12, color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        ax2.grid(False)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        chart2_path = os.path.join(self.visualizations_dir, 'chart2.png')
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"表格2和折线图2已保存到 {self.output_dir} 和 {self.visualizations_dir}")
    
    def generate_report(self) -> str:
        """
        生成分析报告
        
        Returns:
            报告文件路径
        """
        logger.info("开始生成分析报告...")
        
        report_path = os.path.join(self.output_dir, 'analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # 写入报告标题和基本信息
            f.write('# IALM System Post-Hoc Metrics Analysis Report\n\n')
            f.write(f'**Generated Time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'**Analysis Directory**: {self.input_dir}\n')
            f.write(f'**Number of Files Analyzed**: {len(self.ssm_files)}\n\n')
            
            # 写入指标计算结果
            f.write('## 1. Metrics Calculation Results\n\n')
            if self.metrics_results:
                f.write('### 1.1 Basic Metrics\n\n')
                f.write('| Metric Name | Value |\n')
                f.write('|-------------|-------|\n')
                f.write(f"| Total Dialogues | {self.metrics_results.get('total_dialogues', 0)} |\n")
                f.write(f"| Success Count | {self.metrics_results.get('success_count', 0)} |\n")
                f.write(f"| Completed Count | {self.metrics_results.get('complete_count', 0)} |\n")
                f.write(f"| Failed Count | {self.metrics_results.get('failure_count', 0)} |\n")
                f.write(f"| Success Rate | {self.metrics_results.get('success_rate', 0):.2%} |\n")
                f.write(f"| Completed Rate | {self.metrics_results.get('complete_rate', 0):.2%} |\n")
                f.write(f"| Failed Rate | {self.metrics_results.get('failure_rate', 0):.2%} |\n")
                f.write(f"| Average Dialogue Turns | {self.metrics_results.get('average_turns', 0):.2f} |\n")
                f.write(f"| Success Dialogue Average Turns | {self.metrics_results.get('success_avg_turns', 0):.2f} |\n")
                f.write(f"| Completed Dialogue Average Turns | {self.metrics_results.get('complete_avg_turns', 0):.2f} |\n")
                f.write(f"| Success+Completed Dialogue Average Turns | {self.metrics_results.get('success_complete_avg_turns', 0):.2f} |\n")
                f.write(f"| Inform Rate | {self.metrics_results.get('inform_rate', 0):.2%} |\n")
                f.write(f"| BLEU Score | {self.metrics_results.get('bleu', 0):.3f} |\n")
                f.write(f"| Combine Score | {self.metrics_results.get('combine', 0):.3f} |\n\n")
            
            # 写入表格1和折线图1
            f.write('## 2. Grouped Analysis by Experiment Number (20 experiments per group)\n\n')
            f.write('### 2.1 Status Distribution Statistics\n\n')
            
            # 读取并嵌入表格1
            table1_path_md = os.path.join(self.output_dir, 'table1.md')
            if os.path.exists(table1_path_md):
                with open(table1_path_md, 'r', encoding='utf-8') as table_file:
                    # 跳过标题行
                    table_content = ''.join(table_file.readlines()[1:])
                    f.write(table_content)
            
            f.write('\n### 2.2 Status Ratio Trend\n\n')
            chart1_path = os.path.join(self.visualizations_dir, 'chart1.png')
            if os.path.exists(chart1_path):
                f.write(f'![Status Ratio Trend]({os.path.relpath(chart1_path, self.output_dir)})\n\n')
            
            # 写入表格2和折线图2
            f.write('## 3. Grouped Analysis by Domains (10 experiments per group with same domains)\n\n')
            f.write('### 3.1 Status Distribution Statistics\n\n')
            
            # 读取并嵌入表格2
            table2_path_md = os.path.join(self.output_dir, 'table2.md')
            if os.path.exists(table2_path_md):
                with open(table2_path_md, 'r', encoding='utf-8') as table_file:
                    # 跳过标题行
                    table_content = ''.join(table_file.readlines()[1:])
                    f.write(table_content)
            
            f.write('\n### 3.2 Status Ratio Trend\n\n')
            chart2_path = os.path.join(self.visualizations_dir, 'chart2.png')
            if os.path.exists(chart2_path):
                f.write(f"![Status Ratio Trend by Domains]({os.path.relpath(chart2_path, self.output_dir)})\n\n")
            
            # 写入分析和总结
            f.write('## 4. Analysis and Summary\n\n')
            f.write('### 4.1 Key Findings\n\n')
            f.write('1. **Overall Performance Analysis**: The system achieved an overall success rate of {:.2%}, indicating that the system can complete dialogue tasks in most cases.\n'.format(
                self.metrics_results.get('success_rate', 0)
            ))
            f.write('2. **Success/Completed/Failed Distribution**: The system had {} successful dialogues, {} completed dialogues, and {} failed dialogues,\n'.format(
                self.metrics_results.get('success_count', 0),
                self.metrics_results.get('complete_count', 0),
                self.metrics_results.get('failure_count', 0)
            ))
            f.write('   indicating that the system can complete tasks in some cases but may have minor issues.\n')
            f.write('3. **Language Quality Analysis**: The system achieved a BLEU score of {:.3f}, reflecting the quality of generated responses compared to ground truth.\n'.format(
                self.metrics_results.get('bleu', 0)
            ))
            f.write('4. **Information Accuracy**: The system has an inform rate of {:.2%}, demonstrating strong ability to correctly identify user goals and provide appropriate entities.\n'.format(
                self.metrics_results.get('inform_rate', 0)
            ))
            f.write('5. **Combined Performance**: The overall Combine score is {:.3f}, which aggregates Inform, Success, and BLEU metrics,\n'.format(
                self.metrics_results.get('combine', 0)
            ))
            f.write('   providing a comprehensive evaluation of the system\'s overall performance.\n')
            f.write('6. **Trend Analysis**: From the line charts, it can be seen that system performance exhibits {} across different batches,\n'.format(
                'fluctuations' if len(self.experiment_data) > 20 else 'limited data points, making it difficult to determine obvious trends'
            ))
            f.write('   which may be related to experimental conditions or data distribution.\n')
            f.write('7. **Domain Adaptability**: System performance {} across different domains combinations,\n'.format(
                'varies' if len(self.experiment_data) > 10 else 'has limited data, making it difficult to determine obvious differences'
            ))
            f.write('   indicating that the system may perform better in certain domains.\n\n')
            
            f.write('### 4.2 Improvement Recommendations\n\n')
            f.write('1. **Targeted Failure Analysis**: It is recommended to conduct in-depth analysis of failed cases to identify main failure reasons,\n')
            f.write('   with a focus on optimizing relevant modules.\n')
            f.write('2. **Performance Stability**: For batches with large performance fluctuations, analyze their characteristics,\n')
            f.write('   and improve the stability and robustness of the system.\n')
            f.write('3. **Domain Optimization**: For domains with poor performance,\n')
            f.write('   increase data training for relevant domains or optimize domain-specific processing logic.\n')
            f.write('4. **Continuous Monitoring**: It is recommended to establish a continuous monitoring mechanism,\n')
            f.write('   to timely detect system performance changes and respond quickly to issues.\n\n')
            
            f.write('## 5. Appendix\n\n')
            f.write('### 5.1 Analysis Script\n\n')
            f.write('This report was generated by the `src/post_analysis.py` script.\n\n')
            f.write('### 5.2 Visualization Files\n\n')
            f.write('- Table 1: table1.csv\n')
            f.write('- Table 2: table2.csv\n')
            f.write('- Line Chart 1: visualizations/chart1.png\n')
            f.write('- Line Chart 2: visualizations/chart2.png\n\n')
            
            if os.path.exists(os.path.join(self.output_dir, 'detailed_metrics.json')):
                f.write('### 5.3 Detailed Metrics\n\n')
                f.write('Detailed metrics calculation results have been saved to `detailed_metrics.json`.\n')
        
        logger.info(f"分析报告已保存到: {report_path}")
        return report_path
    
    def run(self) -> None:
        """
        运行完整的分析流程
        """
        logger.info("开始事后指标分析...")
        
        # 1. 查找ssm文件
        self.find_ssm_files()
        
        # 2. 解析ssm文件
        self.parse_ssm_files()
        
        # 3. 计算指标
        self.metrics_results = self.calculate_metrics()
        
        # 4. 生成可视化图表
        self.generate_visualizations()
        
        # 5. 生成分析报告
        report_path = self.generate_report()
        
        logger.info(f"事后指标分析完成，报告已保存到: {report_path}")


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="IALM系统事后指标分析脚本，用于分析ssm_XXXX.json实验结果文件"
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="包含ssm_XXXX.json文件的输入目录"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./report",
        help="输出结果目录，默认为当前目录下的report"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别，默认为INFO"
    )
    
    return parser.parse_args()


def main() -> None:
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 确保输入目录存在
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录 {args.input_dir} 不存在")
        sys.exit(1)
    
    # 初始化并运行分析
    analysis = PostAnalysis(args.input_dir, args.output_dir)
    analysis.run()


if __name__ == "__main__":
    main()
