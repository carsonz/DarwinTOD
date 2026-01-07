#!/usr/bin/env python3
"""
IALM结果分析脚本

该脚本用于深入分析IALM系统的实验结果，包括：
1. 策略使用分析
2. 错误模式分析
3. 领域特定分析
4. 对话流分析

使用方法:
    python experiments/analysis.py --result_dir experiments/results/ialm_experiment_20231201_120000
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(log_level: str = "INFO") -> None:
    """设置日志记录"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_results(result_dir: str) -> Dict[str, Any]:
    """加载实验结果"""
    results_path = os.path.join(result_dir, "final_results.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def analyze_strategy_usage(results: Dict[str, Any], output_dir: str) -> None:
    """分析策略使用情况"""
    dialogues = results.get("results", [])
    
    # 收集所有使用的策略
    all_strategies = []
    agent_strategies = defaultdict(list)
    
    for dialogue in dialogues:
        for turn in dialogue.get("dialogue", {}).get("turns", []):
            for agent_name, agent_data in turn.items():
                if isinstance(agent_data, dict) and "used_strategies" in agent_data:
                    strategies = agent_data.get("used_strategies", [])
                    all_strategies.extend(strategies)
                    agent_strategies[agent_name].extend(strategies)
    
    # 统计策略使用频率
    strategy_counts = Counter(all_strategies)
    
    # 创建分析目录
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 保存策略使用统计
    strategy_usage_path = os.path.join(analysis_dir, "strategy_usage.csv")
    strategy_df = pd.DataFrame(list(strategy_counts.items()), columns=["Strategy", "Count"])
    strategy_df.to_csv(strategy_usage_path, index=False)
    
    # 可视化策略使用情况
    plt.figure(figsize=(12, 8))
    top_strategies = strategy_df.head(20)  # 只显示前20个最常用的策略
    
    bars = plt.barh(top_strategies["Strategy"], top_strategies["Count"])
    plt.title('策略使用频率 (Top 20)')
    plt.xlabel('使用次数')
    plt.ylabel('策略')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'strategy_usage.png'), dpi=300)
    plt.close()
    
    # 按Agent分析策略使用
    agent_strategy_data = []
    for agent_name, strategies in agent_strategies.items():
        agent_strategy_counts = Counter(strategies)
        for strategy, count in agent_strategy_counts.items():
            agent_strategy_data.append({
                "Agent": agent_name,
                "Strategy": strategy,
                "Count": count
            })
    
    if agent_strategy_data:
        agent_strategy_df = pd.DataFrame(agent_strategy_data)
        agent_strategy_path = os.path.join(analysis_dir, "agent_strategy_usage.csv")
        agent_strategy_df.to_csv(agent_strategy_path, index=False)
        
        # 可视化各Agent的策略使用情况
        plt.figure(figsize=(14, 10))
        top_agent_strategies = agent_strategy_df.groupby("Agent").apply(
            lambda x: x.nlargest(10, "Count")
        ).reset_index(drop=True)
        
        sns.barplot(data=top_agent_strategies, x="Count", y="Strategy", hue="Agent")
        plt.title('各Agent策略使用情况 (Top 10 per Agent)')
        plt.xlabel('使用次数')
        plt.ylabel('策略')
        plt.legend(title='Agent')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'agent_strategy_usage.png'), dpi=300)
        plt.close()


def analyze_error_patterns(results: Dict[str, Any], output_dir: str) -> None:
    """分析错误模式"""
    dialogues = results.get("results", [])
    
    # 收集错误信息
    errors = []
    failed_dialogues = []
    
    for dialogue in dialogues:
        dialogue_id = dialogue.get("dialogue_id", "unknown")
        domain = dialogue.get("dialogue", {}).get("domain", "unknown")
        
        # 检查对话是否成功
        success = dialogue.get("success", False)
        if not success:
            failed_dialogues.append({
                "dialogue_id": dialogue_id,
                "domain": domain
            })
        
        # 收集错误信息
        for turn in dialogue.get("dialogue", {}).get("turns", []):
            for agent_name, agent_data in turn.items():
                if isinstance(agent_data, dict) and "error" in agent_data:
                    errors.append({
                        "dialogue_id": dialogue_id,
                        "domain": domain,
                        "agent": agent_name,
                        "error": agent_data.get("error"),
                        "turn_id": turn.get("turn_id", "unknown")
                    })
    
    # 创建分析目录
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 保存错误信息
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_path = os.path.join(analysis_dir, "errors.csv")
        errors_df.to_csv(errors_path, index=False)
        
        # 分析错误类型
        error_types = errors_df["error"].value_counts()
        
        plt.figure(figsize=(12, 8))
        error_types.head(10).plot(kind='bar')
        plt.title('错误类型分布 (Top 10)')
        plt.xlabel('错误类型')
        plt.ylabel('出现次数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'error_types.png'), dpi=300)
        plt.close()
        
        # 按Agent分析错误
        agent_errors = errors_df.groupby("agent")["error"].value_counts().unstack(fill_value=0)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(agent_errors, annot=True, fmt="d", cmap="YlGnBu")
        plt.title('各Agent错误类型分布')
        plt.xlabel('错误类型')
        plt.ylabel('Agent')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'agent_errors.png'), dpi=300)
        plt.close()
    
    # 分析失败对话
    if failed_dialogues:
        failed_df = pd.DataFrame(failed_dialogues)
        failed_path = os.path.join(analysis_dir, "failed_dialogues.csv")
        failed_df.to_csv(failed_path, index=False)
        
        # 按领域分析失败对话
        domain_failures = failed_df["domain"].value_counts()
        
        plt.figure(figsize=(12, 8))
        domain_failures.plot(kind='bar')
        plt.title('各领域失败对话数量')
        plt.xlabel('领域')
        plt.ylabel('失败对话数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'domain_failures.png'), dpi=300)
        plt.close()


def analyze_domain_performance(results: Dict[str, Any], output_dir: str) -> None:
    """分析各领域性能"""
    dialogues = results.get("results", [])
    
    # 按领域分组对话
    domain_dialogues = defaultdict(list)
    for dialogue in dialogues:
        domain = dialogue.get("dialogue", {}).get("domain", "unknown")
        domain_dialogues[domain].append(dialogue)
    
    # 计算各领域指标
    domain_metrics = []
    for domain, ds in domain_dialogues.items():
        success_count = sum(1 for d in ds if d.get("success", False))
        inform_count = sum(1 for d in ds if d.get("inform_all", False))
        
        # 计算BLEU分数（如果有）
        bleu_scores = [d.get("bleu", 0) for d in ds if "bleu" in d]
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        
        # 计算对话长度
        turn_counts = [len(d.get("dialogue", {}).get("turns", [])) for d in ds]
        avg_turns = np.mean(turn_counts) if turn_counts else 0
        
        domain_metrics.append({
            "Domain": domain,
            "Dialogues": len(ds),
            "Success Rate": success_count / len(ds),
            "Inform Rate": inform_count / len(ds),
            "Avg BLEU": avg_bleu,
            "Avg Turns": avg_turns
        })
    
    # 创建分析目录
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 保存领域指标
    domain_df = pd.DataFrame(domain_metrics)
    domain_path = os.path.join(analysis_dir, "domain_metrics.csv")
    domain_df.to_csv(domain_path, index=False)
    
    # 可视化领域性能
    metrics_to_plot = ["Success Rate", "Inform Rate", "Avg BLEU"]
    
    plt.figure(figsize=(14, 8))
    domain_df_melted = domain_df.melt(id_vars=["Domain"], value_vars=metrics_to_plot, 
                                     var_name="Metric", value_name="Value")
    
    sns.barplot(data=domain_df_melted, x="Domain", y="Value", hue="Metric")
    plt.title('各领域性能指标')
    plt.xlabel('领域')
    plt.ylabel('值')
    plt.xticks(rotation=45)
    plt.legend(title='指标')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'domain_performance.png'), dpi=300)
    plt.close()


def analyze_dialogue_flow(results: Dict[str, Any], output_dir: str) -> None:
    """分析对话流"""
    dialogues = results.get("results", [])
    
    # 收集对话流数据
    dialogue_flows = []
    for dialogue in dialogues:
        dialogue_id = dialogue.get("dialogue_id", "unknown")
        domain = dialogue.get("dialogue", {}).get("domain", "unknown")
        turns = dialogue.get("dialogue", {}).get("turns", [])
        
        for i, turn in enumerate(turns):
            turn_id = turn.get("turn_id", i)
            
            # 分析每个Agent的响应
            for agent_name, agent_data in turn.items():
                if isinstance(agent_data, dict):
                    # 获取响应长度
                    response = agent_data.get("response", "")
                    response_length = len(response.split()) if response else 0
                    
                    # 获取使用的策略
                    strategies = agent_data.get("used_strategies", [])
                    
                    dialogue_flows.append({
                        "dialogue_id": dialogue_id,
                        "domain": domain,
                        "turn_id": turn_id,
                        "agent": agent_name,
                        "response_length": response_length,
                        "num_strategies": len(strategies)
                    })
    
    # 创建分析目录
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 保存对话流数据
    flow_df = pd.DataFrame(dialogue_flows)
    flow_path = os.path.join(analysis_dir, "dialogue_flow.csv")
    flow_df.to_csv(flow_path, index=False)
    
    # 分析响应长度随轮次的变化
    plt.figure(figsize=(12, 8))
    for agent in flow_df["agent"].unique():
        agent_data = flow_df[flow_df["agent"] == agent]
        turn_lengths = agent_data.groupby("turn_id")["response_length"].mean()
        plt.plot(turn_lengths.index, turn_lengths.values, label=agent, marker='o')
    
    plt.title('各Agent响应长度随轮次变化')
    plt.xlabel('轮次')
    plt.ylabel('平均响应长度（词数）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'response_length_by_turn.png'), dpi=300)
    plt.close()
    
    # 分析策略使用随轮次的变化
    plt.figure(figsize=(12, 8))
    for agent in flow_df["agent"].unique():
        agent_data = flow_df[flow_df["agent"] == agent]
        turn_strategies = agent_data.groupby("turn_id")["num_strategies"].mean()
        plt.plot(turn_strategies.index, turn_strategies.values, label=agent, marker='o')
    
    plt.title('各Agent策略使用数量随轮次变化')
    plt.xlabel('轮次')
    plt.ylabel('平均策略使用数量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'strategies_by_turn.png'), dpi=300)
    plt.close()


def generate_analysis_report(results: Dict[str, Any], output_dir: str) -> str:
    """生成分析报告"""
    report_path = os.path.join(output_dir, "analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# IALM系统结果分析报告\n\n")
        
        # 策略使用分析
        f.write("## 策略使用分析\n\n")
        f.write("策略使用情况的分析结果请参见 `analysis/strategy_usage.png` 和 `analysis/agent_strategy_usage.png`。\n\n")
        
        # 错误模式分析
        f.write("## 错误模式分析\n\n")
        f.write("错误模式的分析结果请参见 `analysis/error_types.png` 和 `analysis/agent_errors.png`。\n\n")
        
        # 领域性能分析
        f.write("## 领域性能分析\n\n")
        f.write("各领域性能的分析结果请参见 `analysis/domain_performance.png`。\n\n")
        
        # 对话流分析
        f.write("## 对话流分析\n\n")
        f.write("对话流的分析结果请参见 `analysis/response_length_by_turn.png` 和 `analysis/strategies_by_turn.png`。\n\n")
    
    return report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析IALM实验结果")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="实验结果目录路径")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="日志级别")
    parser.add_argument("--strategy_analysis", action="store_true", default=True,
                        help="进行策略使用分析")
    parser.add_argument("--error_analysis", action="store_true", default=True,
                        help="进行错误模式分析")
    parser.add_argument("--domain_analysis", action="store_true", default=True,
                        help="进行领域性能分析")
    parser.add_argument("--flow_analysis", action="store_true", default=True,
                        help="进行对话流分析")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 检查结果目录
    if not os.path.exists(args.result_dir):
        logger.error(f"结果目录不存在: {args.result_dir}")
        return
    
    # 加载实验结果
    logger.info(f"加载实验结果: {args.result_dir}")
    results = load_results(args.result_dir)
    
    # 创建分析目录
    analysis_dir = os.path.join(args.result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 进行各种分析
    if args.strategy_analysis:
        logger.info("进行策略使用分析...")
        analyze_strategy_usage(results, args.result_dir)
    
    if args.error_analysis:
        logger.info("进行错误模式分析...")
        analyze_error_patterns(results, args.result_dir)
    
    if args.domain_analysis:
        logger.info("进行领域性能分析...")
        analyze_domain_performance(results, args.result_dir)
    
    if args.flow_analysis:
        logger.info("进行对话流分析...")
        analyze_dialogue_flow(results, args.result_dir)
    
    # 生成分析报告
    logger.info("生成分析报告...")
    report_path = generate_analysis_report(results, args.result_dir)
    logger.info(f"分析报告已保存到: {report_path}")
    
    logger.info("分析完成")


if __name__ == "__main__":
    main()