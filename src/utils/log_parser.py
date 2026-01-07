"""
日志文件解析器

专门用于解析IALM实验结果中的日志文件，包括：
1. multiwoz_dialogue_metrics.json - 对话指标文件
2. ssm_*.json - 对话详细过程文件

提供统一的接口进行数据解析和提取。
"""

import json
import os
import glob
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class LogParser:
    """日志文件解析器"""
    
    def __init__(self):
        """初始化解析器"""
        pass
    
    def parse_metrics_file(self, file_path: str) -> Dict[str, Any]:
        """
        解析对话指标文件 (multiwoz_dialogue_metrics.json)
        
        Args:
            file_path: 指标文件路径
            
        Returns:
            解析后的指标数据
        """
        logger.info(f"解析对话指标文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            raise
        
        # 提取总体统计信息
        total_dialogues = data.get("total_dialogues", 0)
        summary = {
            "total_dialogues": total_dialogues,
            "success_count": data.get("success_count", 0),
            "complete_count": data.get("complete_count", 0),
            "failure_count": data.get("failure_count", 0),
            "total_turns": data.get("total_turns", 0),
            "total_time": data.get("total_time", 0.0),
            "success_rate": data.get("success_rate", 0.0),
            "average_turns": data.get("total_turns", 0) / max(total_dialogues, 1),
            "average_time": data.get("total_time", 0.0) / max(total_dialogues, 1)
        }
        
        # 提取每个对话的详细信息
        dialogue_details = []
        for detail in data.get("details", []):
            dialogue_info = {
                "dialogue_id": detail.get("dialogue_id"),
                "domain": detail.get("domain", []),
                "goal": detail.get("goal", {}),
                "final_state": detail.get("final_state"),
                "total_turns": detail.get("total_turns", 0),
                "total_time": detail.get("total_time", 0.0),
                "belief_state": detail.get("belief_state", {})
            }
            dialogue_details.append(dialogue_info)
        
        logger.info(f"成功解析 {len(dialogue_details)} 个对话的指标数据")
        
        return {
            "summary": summary,
            "dialogue_details": dialogue_details
        }
    
    def parse_ssm_file(self, file_path: str) -> Dict[str, Any]:
        """
        解析SSM对话数据文件 (ssm_*.json)
        
        Args:
            file_path: SSM文件路径
            
        Returns:
            解析后的SSM数据
        """
        logger.info(f"解析SSM对话数据文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            raise
        
        dialogues = data.get("dialogues", {})
        dialogue_id = dialogues.get("dialogue_id")
        domain = dialogues.get("domain", [])
        goal = dialogues.get("goal", {})
        turns = dialogues.get("turns", [])
        
        # 解析每轮对话数据
        turn_details = []
        all_used_strategies = []
        agent_prompts = defaultdict(list)
        agent_results = defaultdict(list)
        
        for turn in turns:
            turn_num = turn.get("turn_num")
            turn_info = {
                "turn_num": turn_num,
                "user": turn.get("user"),
                "system": turn.get("system"),
                "belief_state": turn.get("belief_state", {}),
                "system_action": turn.get("system_action"),
                "db_results": turn.get("db_results", {}),
                "agent_data": turn.get("agent_data", {})
            }
            turn_details.append(turn_info)
            
            # 收集该轮使用的HSM策略和其他代理数据
            agent_data = turn.get("agent_data", {})
            for agent, agent_info in agent_data.items():
                # 收集prompt和结果
                if "prompt" in agent_info:
                    agent_prompts[agent].append({
                        "turn_num": turn_num,
                        "prompt": agent_info["prompt"]
                    })
                
                if "result" in agent_info:
                    agent_results[agent].append({
                        "turn_num": turn_num,
                        "result": agent_info["result"]
                    })
                
                # 收集HSM策略信息
                hsm_strategies = agent_info.get("hsm", [])
                for strategy in hsm_strategies:
                    strategy_info = {
                        "turn_num": turn_num,
                        "agent": agent,
                        "strategy_id": strategy.get("id"),
                        "strategy_type": strategy.get("type"),
                        "content": strategy.get("content", ""),
                        "similarity_score": strategy.get("similarity_score", 0.0),
                        "metadata": strategy.get("metadata", {}),
                        "applicable_agents": strategy.get("applicable_agents", []),
                        "scope": strategy.get("scope", ""),
                        "domain": strategy.get("domain", "")
                    }
                    all_used_strategies.append(strategy_info)
        
        logger.info(f"成功解析 {len(turns)} 轮对话数据，收集到 {len(all_used_strategies)} 个策略使用记录")
        
        return {
            "dialogue_info": {
                "dialogue_id": dialogue_id,
                "domain": domain,
                "goal": goal,
                "total_turns": len(turns)
            },
            "turn_details": turn_details,
            "used_strategies": all_used_strategies,
            "agent_prompts": dict(agent_prompts),
            "agent_results": dict(agent_results)
        }
    
    def parse_experiment_results(self, results_dir: str) -> Dict[str, Any]:
        """
        解析完整实验结果目录
        
        Args:
            results_dir: 实验结果目录路径
            
        Returns:
            完整的实验结果数据
        """
        logger.info(f"解析实验结果目录: {results_dir}")
        
        # 1. 解析对话指标文件
        metrics_file = os.path.join(results_dir, "multiwoz_dialogue_metrics.json")
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"找不到指标文件: {metrics_file}")
        
        metrics_data = self.parse_metrics_file(metrics_file)
        
        # 2. 解析所有SSM文件
        ssm_files = glob.glob(os.path.join(results_dir, "ssm_*.json"))
        ssm_data_list = []
        
        for ssm_file in ssm_files:
            try:
                ssm_data = self.parse_ssm_file(ssm_file)
                ssm_data_list.append(ssm_data)
            except Exception as e:
                logger.warning(f"跳过SSM文件 {ssm_file}: {e}")
        
        # 3. 构建完整的实验结果
        combined_results = []
        
        # 合并metrics和SSM数据
        for metrics_detail in metrics_data["dialogue_details"]:
            dialogue_id = metrics_detail["dialogue_id"]
            
            # 找到对应的SSM数据
            ssm_data = None
            for ssm in ssm_data_list:
                if ssm["dialogue_info"]["dialogue_id"] == dialogue_id:
                    ssm_data = ssm
                    break
            
            # 合并数据
            combined_dialogue = {
                **metrics_detail,  # 来自metrics的对话信息
                "ssm_data": ssm_data  # SSM详细数据
            }
            combined_results.append(combined_dialogue)
        
        logger.info(f"成功合并 {len(combined_results)} 个对话的完整数据")
        
        return {
            "summary": metrics_data["summary"],
            "dialogue_results": combined_results,
            "total_ssm_files": len(ssm_data_list),
            "parsed_at": logger.handlers[0].formatter.formatTime(
                logger.makeRecord(
                    logger.name, logging.INFO, "", 0, 
                    "Parsed at this time", (), None
                )
            ) if logger.handlers else "N/A"
        }
    
    def analyze_strategy_usage(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析策略使用情况
        
        Args:
            experiment_results: 实验结果数据
            
        Returns:
            策略使用统计信息
        """
        logger.info("分析策略使用情况...")
        
        dialogue_results = experiment_results["dialogue_results"]
        strategy_stats = defaultdict(lambda: {
            "total_usage": 0,
            "successful_usage": 0,
            "agents": set(),
            "turns": [],
            "domains": set(),
            "types": set()
        })
        
        for dialogue in dialogue_results:
            final_state = dialogue.get("final_state")
            is_successful = final_state == "success"
            ssm_data = dialogue.get("ssm_data", {})
            
            if ssm_data and "used_strategies" in ssm_data:
                for strategy_info in ssm_data["used_strategies"]:
                    strategy_id = strategy_info["strategy_id"]
                    
                    stats = strategy_stats[strategy_id]
                    stats["total_usage"] += 1
                    stats["agents"].add(strategy_info["agent"])
                    stats["turns"].append(strategy_info["turn_num"])
                    stats["domains"].update(dialogue.get("domain", []))
                    stats["types"].add(strategy_info["strategy_type"])
                    
                    if is_successful:
                        stats["successful_usage"] += 1
        
        # 转换为可序列化格式
        strategy_analysis = {}
        for strategy_id, stats in strategy_stats.items():
            turns = stats["turns"]
            strategy_analysis[strategy_id] = {
                "total_usage": stats["total_usage"],
                "successful_usage": stats["successful_usage"],
                "success_rate": stats["successful_usage"] / stats["total_usage"] if stats["total_usage"] > 0 else 0.0,
                "agents": list(stats["agents"]),
                "domains": list(stats["domains"]),
                "types": list(stats["types"]),
                "avg_turn_number": sum(turns) / len(turns) if turns else 0.0,
                "min_turn_number": min(turns) if turns else 0,
                "max_turn_number": max(turns) if turns else 0
            }
        
        logger.info(f"分析了 {len(strategy_analysis)} 个不同策略的使用情况")
        
        return strategy_analysis
    
    def extract_dialogue_details(self, dialogue_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取单个对话的详细信息
        
        Args:
            dialogue_result: 单个对话的结果数据
            
        Returns:
            详细分析信息
        """
        dialogue_id = dialogue_result.get("dialogue_id")
        final_state = dialogue_result.get("final_state")
        ssm_data = dialogue_result.get("ssm_data")
        
        detail = {
            "dialogue_id": dialogue_id,
            "final_state": final_state,
            "domain": dialogue_result.get("domain", []),
            "goal": dialogue_result.get("goal", {}),
            "total_turns": dialogue_result.get("total_turns", 0),
            "total_time": dialogue_result.get("total_time", 0.0)
        }
        
        if ssm_data:
            # 提取使用的策略信息
            used_strategies = ssm_data.get("used_strategies", [])
            detail.update({
                "strategy_count": len(used_strategies),
                "strategies_by_agent": defaultdict(int),
                "strategy_types": defaultdict(int),
                "strategies_used": []
            })
            
            for strategy in used_strategies:
                detail["strategies_by_agent"][strategy["agent"]] += 1
                detail["strategy_types"][strategy["strategy_type"]] += 1
                
                detail["strategies_used"].append({
                    "id": strategy["strategy_id"],
                    "agent": strategy["agent"],
                    "type": strategy["strategy_type"],
                    "turn": strategy["turn_num"],
                    "similarity_score": strategy["similarity_score"]
                })
            
            # 转换为普通dict
            detail["strategies_by_agent"] = dict(detail["strategies_by_agent"])
            detail["strategy_types"] = dict(detail["strategy_types"])
            
            # 提取代理交互信息
            agent_prompts = ssm_data.get("agent_prompts", {})
            agent_results = ssm_data.get("agent_results", {})
            
            detail.update({
                "agent_interactions": {
                    "prompts_by_agent": {k: len(v) for k, v in agent_prompts.items()},
                    "results_by_agent": {k: len(v) for k, v in agent_results.items()}
                }
            })
        
        return detail
    
    def generate_usage_report(self, experiment_results: Dict[str, Any]) -> str:
        """
        生成使用情况报告
        
        Args:
            experiment_results: 实验结果数据
            
        Returns:
            格式化的报告文本
        """
        strategy_analysis = self.analyze_strategy_usage(experiment_results)
        dialogue_results = experiment_results["dialogue_results"]
        
        report = []
        report.append("# 实验结果分析报告\n")
        
        # 总体统计
        summary = experiment_results["summary"]
        report.append("## 总体统计")
        report.append(f"- 总对话数: {summary['total_dialogues']}")
        report.append(f"- 成功对话数: {summary['success_count']}")
        report.append(f"- 成功率: {summary['success_rate']:.2%}")
        report.append(f"- 平均轮数: {summary['average_turns']:.1f}")
        report.append(f"- 平均时间: {summary['average_time']:.1f}秒")
        report.append("")
        
        # 策略分析
        report.append("## 策略使用分析")
        report.append(f"- 使用了 {len(strategy_analysis)} 个不同策略")
        
        # 最常用的策略
        top_strategies = sorted(
            strategy_analysis.items(),
            key=lambda x: x[1]["total_usage"],
            reverse=True
        )[:10]
        
        report.append("\n### 最常用的策略 (Top 10)")
        report.append("| 策略ID | 使用次数 | 成功率 | 代理类型 |")
        report.append("|--------|----------|--------|----------|")
        
        for strategy_id, stats in top_strategies:
            agents = ", ".join(stats["agents"])
            report.append(f"| {strategy_id} | {stats['total_usage']} | {stats['success_rate']:.2%} | {agents} |")
        
        # 按代理类型分析
        agent_usage = defaultdict(int)
        for stats in strategy_analysis.values():
            for agent in stats["agents"]:
                agent_usage[agent] += stats["total_usage"]
        
        report.append(f"\n### 各代理策略使用次数")
        for agent, count in sorted(agent_usage.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {agent}: {count} 次")
        
        return "\n".join(report)


def main():
    """主函数 - 用于测试解析功能"""
    parser = LogParser()
    
    # 这里可以添加测试代码
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "results", "multiwoz_1_20251112_121602")
    
    if os.path.exists(results_dir):
        try:
            results = parser.parse_experiment_results(results_dir)
            print("解析成功！")
            print(f"共解析 {results['summary']['total_dialogues']} 个对话")
            
            # 生成报告
            report = parser.generate_usage_report(results)
            print(report)
            
        except Exception as e:
            print(f"解析失败: {e}")
    else:
        print(f"结果目录不存在: {results_dir}")


if __name__ == "__main__":
    main()