#!/usr/bin/env python3
"""
IALM实验运行脚本

该脚本采用面向对象设计原则，实现高内聚低耦合的架构，用于运行IALM系统的实验。
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


class ExperimentConfig:
    """
    实验配置类，负责加载和管理配置
    """
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """
        初始化配置类
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔符访问嵌套配置
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def save(self, output_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)


class ExperimentLogger:
    """
    实验日志类，负责配置和管理日志
    """
    
    def __init__(self, log_level: str = "INFO", log_file: str = None):
        """
        初始化日志类
        
        Args:
            log_level: 日志级别
            log_file: 日志文件路径
        """
        self.log_level = log_level.upper()
        self.log_file = log_file
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """
        设置日志记录
        """
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        handlers = [logging.StreamHandler()]
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            handlers.append(logging.FileHandler(self.log_file))
        
        # 设置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 设置所有处理器的格式
        for handler in handlers:
            handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(handler)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        获取日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            日志记录器实例
        """
        return logging.getLogger(name)


class ExperimentOutput:
    """
    实验输出类，负责管理输出目录和文件
    """
    
    def __init__(self, base_dir: str, experiment_name: str):
        """
        初始化输出类
        
        Args:
            base_dir: 基础输出目录
            experiment_name: 实验名称
        """
        self.base_dir = base_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self._create_output_dir()
    
    def _create_output_dir(self) -> str:
        """
        创建输出目录
        
        Returns:
            输出目录路径
        """
        output_dir = os.path.join(self.base_dir, f"{self.experiment_name}_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def get_output_path(self, filename: str) -> str:
        """
        获取输出文件路径
        
        Args:
            filename: 文件名
            
        Returns:
            完整的输出文件路径
        """
        return os.path.join(self.output_dir, filename)
    
    def save_results(self, results: Any, filename: str) -> None:
        """
        保存结果到文件
        
        Args:
            results: 结果数据
            filename: 文件名
        """
        file_path = self.get_output_path(filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)


class LLMClientFactory:
    """
    LLM客户端工厂类，负责创建和管理LLM客户端
    """
    
    @staticmethod
    def create_llm_client(provider: str = "vllmLocal", config_path: str = "config/default_config.yaml") -> Any:
        """
        创建LLM客户端
        
        Args:
            provider: LLM提供商
            config_path: 配置文件路径
            
        Returns:
            LLM客户端实例
        """
        from src.utils.llm_client import LLMClient
        return LLMClient(provider=provider, config_path=config_path)


class ExperimentRunner:
    """
    实验运行器类，负责运行实验的核心逻辑
    """
    
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger, output: ExperimentOutput):
        """
        初始化实验运行器
        
        Args:
            config: 实验配置实例
            logger: 实验日志实例
            output: 实验输出实例
        """
        self.config = config
        self.logger = logger.get_logger(__name__)
        self.output = output
        self.llm_client = None
        self.dialogue_manager = None
    
    def _initialize_components(self) -> None:
        """
        初始化实验组件
        """
        # 初始化LLM客户端
        self.logger.info("初始化LLM客户端...")
        self.llm_client = LLMClientFactory.create_llm_client(
            provider="vllmLocal",
            config_path=self.config.config_path
        )
        
        # 初始化对话管理器
        self.logger.info("初始化对话管理器...")
        from src.core.dialogue_manager1 import DialogueManager1
        self.dialogue_manager = DialogueManager1(self.config.config, self.llm_client, self.output.output_dir)
    
    def run_dialogue_experiment(self, dataset: str, num_dialogues: int) -> Dict[str, Any]:
        """
        运行对话实验
        
        Args:
            dataset: 数据集名称
            num_dialogues: 对话数量
            
        Returns:
            实验结果
        """
        self.logger.info(f"开始运行 {dataset} 数据集实验，共 {num_dialogues} 个对话...")
        
        # 初始化组件
        self._initialize_components()
        
        # 加载对话目标
        self.logger.info(f"加载 {num_dialogues} 个对话目标...")
        goals, domains, ids = self.dialogue_manager.load_dialogue_goals(dataset, num_dialogues, data_split="test")
        
        # 运行对话
        results = []
        success_count = 0
        complete_count = 0
        failure_count = 0
        
        for i, goal in enumerate(tqdm(goals, desc=f"Running {dataset} dialogues", unit="dialogue")):
            dialogue_id = ids[i]
            domain = domains[i]
            self.logger.info(f"运行对话 {i+1}/{num_dialogues}: {dialogue_id}")
            
            try:
                # 运行单个对话
                result = self.dialogue_manager.run_dialogue(dialogue_id, domain, goal)
                results.append(result)
                
                # 统计结果
                if result["final_state"] in ["success"]:
                    success_count += 1
                elif result["final_state"] in ["completed"]:
                    complete_count += 1
                else:
                    failure_count += 1
                    
                self.logger.info(f"对话 {dialogue_id} 完成，状态: {result['final_state']}, "
                              f"轮次: {result['total_turns']}, 时间: {result['total_time']:.2f}s")
            except Exception as e:
                import traceback
                self.logger.error(f"运行对话 {dialogue_id} 出错: {str(e)}")
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                failure_count += 1
                results.append({
                    "dialogue_id": dialogue_id,
                    "goal": goal,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "final_state": "error",
                    "total_turns": 0,
                    "total_time": 0,
                    "turns": []
                })
        
        # 计算基础统计信息
        basic_metrics = {
            "total_dialogues": num_dialogues,
            "success_count": success_count,
            "complete_count": complete_count,
            "failure_count": failure_count,
            "success_rate": success_count / num_dialogues if num_dialogues > 0 else 0,
            "complete_rate": complete_count / num_dialogues if num_dialogues > 0 else 0,
            "details": results
        }
        
        # 保存基础统计信息
        self.output.save_results(basic_metrics, f"{dataset}_dialogue_metrics.json")
        
        # 尝试计算详细评估指标
        final_metrics = self._calculate_detailed_metrics(results, basic_metrics)
        
        # 保存最终结果
        self.output.save_results(final_metrics, "final_evaluation_results.json")
        
        self.logger.info(f"实验完成: {success_count}/{num_dialogues} 个对话成功")
        self.logger.info(f"结果已保存到: {self.output.output_dir}")
        
        return {
            "results": results,
            "metrics": final_metrics,
            "output_dir": self.output.output_dir
        }
    
    def _calculate_detailed_metrics(self, results: Any, basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算详细评估指标
        
        Args:
            results: 对话结果
            basic_metrics: 基础统计信息
            
        Returns:
            包含详细指标的字典
        """
        self.logger.info("开始计算详细评估指标...")
        
        try:
            from src.utils.metrics import MetricsCalculator
            from src.utils.evaluation_report_generator import EvaluationReportGenerator
            from src.utils.ssm_reader import SSMDataReader
            
            # 1. 从SSM文件中读取数据
            ssm_reader = SSMDataReader()
            ssm_data_list = ssm_reader.read_all_ssm_files(self.output.output_dir)
            
            # 2. 将SSM数据转换为metrics_calculator所需的格式
            # 如果有SSM数据，使用SSM数据；否则使用原始results
            if ssm_data_list:
                self.logger.info("使用从SSM文件中读取的数据计算指标")
                # 将SSM数据转换为与原来results格式兼容的数据结构
                converted_results = []
                for ssm_data in ssm_data_list:
                    dialogue_info = ssm_data['dialogue_info']
                    turns = ssm_data['turns']
                    raw_data = ssm_data['raw_data']
                    
                    # 查找对应的原始结果
                    original_result = next((r for r in results if r['dialogue_id'] == dialogue_info['dialogue_id']), None)
                    
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
                            "ssm_data": ssm_data  # 添加ssm数据到结果中
                        }
                        converted_results.append(converted_result)
                    else:
                        # 如果没有原始结果，创建一个新的
                        converted_result = {
                            "dialogue_id": dialogue_info['dialogue_id'],
                            "final_state": raw_data['metadata']['final_state'] if raw_data and 'metadata' in raw_data else "unknown",
                            "total_turns": dialogue_info['total_turns'],
                            "total_time": raw_data['metadata']['total_time'] if raw_data and 'metadata' in raw_data else 0,
                            "goal": goal,
                            "turns": turns,
                            "ssm_data": ssm_data  # 添加ssm数据到结果中
                        }
                        converted_results.append(converted_result)
                
                # 使用转换后的结果
                processed_results = converted_results
            else:
                self.logger.info("没有找到SSM文件，使用原始results数据计算指标")
                processed_results = results
            
            # 3. 计算详细评估指标
            metrics_calculator = MetricsCalculator(self.config.config)
            all_metrics = metrics_calculator.calculate_all_metrics(processed_results, self.dialogue_manager)
            
            # 4. 合并基础指标和详细指标
            final_metrics = {**basic_metrics, **all_metrics}
            
            # 5. 生成评估报告
            self.logger.info("生成评估报告...")
            report_generator = EvaluationReportGenerator(self.config.config)
            report_dir = report_generator.generate_comprehensive_report(
                processed_results, all_metrics, self.output.output_dir
            )
            
            # 6. 保存详细指标到单独文件
            self.output.save_results(all_metrics, "detailed_metrics.json")
            
            self.logger.info(f"评估完成，报告已保存到: {report_dir}")
            
        except ImportError as e:
            self.logger.warning(f"评估模块导入失败，仅使用基础指标: {e}")
            final_metrics = basic_metrics
        except Exception as e:
            import traceback
            self.logger.error(f"计算详细评估指标出错: {str(e)}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            final_metrics = basic_metrics
        
        return final_metrics


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    # 简化的命令行参数处理，支持 "python run.py multiwoz 10" 格式
    if len(sys.argv) == 3:
        parser = argparse.ArgumentParser(description="运行IALM对话实验")
        parser.add_argument("dataset", type=str, choices=["multiwoz", "sgd"],
                            help="数据集名称")
        parser.add_argument("num_dialogues", type=int,
                            help="运行的对话数量")
        parser.add_argument("--config", type=str, default="config/default_config.yaml",
                            help="配置文件路径")
        parser.add_argument("--output_dir", type=str, default="experiments/results",
                            help="输出目录")
        parser.add_argument("--log_level", type=str, default="INFO",
                            help="日志级别")
        
        # 创建一个命名空间对象并设置参数
        args = argparse.Namespace()
        args.dataset = sys.argv[1]
        args.num_dialogues = int(sys.argv[2])
        args.config = "config/default_config.yaml"
        args.output_dir = "experiments/results"
        args.log_level = "INFO"
    else:
        # 完整的命令行参数处理
        parser = argparse.ArgumentParser(description="运行IALM对话实验")
        parser.add_argument("dataset", type=str, choices=["multiwoz", "sgd"],
                            help="数据集名称 (multiwoz 或 sgd)")
        parser.add_argument("num_dialogues", type=int,
                            help="运行的对话数量")
        parser.add_argument("--config", type=str, default="config/default_config.yaml",
                            help="配置文件路径")
        parser.add_argument("--output_dir", type=str, default="experiments/results",
                            help="输出目录")
        parser.add_argument("--log_level", type=str, default="INFO",
                            help="日志级别")
        
        args = parser.parse_args()
    
    return args


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建配置实例
    config = ExperimentConfig(args.config)
    
    # 创建输出目录
    experiment_name = f"{args.dataset}_{args.num_dialogues}"
    output = ExperimentOutput(args.output_dir, experiment_name)
    
    # 创建日志实例
    log_file = output.get_output_path("experiment.log")
    logger = ExperimentLogger(args.log_level, log_file)
    
    # 保存配置到输出目录
    config.save(output.get_output_path("config.yaml"))
    
    # 创建实验运行器
    experiment_runner = ExperimentRunner(config, logger, output)
    
    # 运行实验
    try:
        results = experiment_runner.run_dialogue_experiment(
            dataset=args.dataset,
            num_dialogues=args.num_dialogues
        )
        
        logger.get_logger().info("实验脚本执行完毕")
        
    except Exception as e:
        logger.get_logger().error(f"实验运行出错: {str(e)}")
        import traceback
        logger.get_logger().error(f"详细错误信息: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
