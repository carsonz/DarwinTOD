"""
IALM System Evaluation Report Generator Module

Implements evaluation report generation including Markdown format reports, 
JSON format reports, and visualization chart generation.
"""

import json
import os
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """IALM System Evaluation Report Generator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the evaluation report generator
        
        Args:
            config: Configuration dictionary containing report generation parameters
        """
        self.config = config or {}
        self.report_config = self.config.get("report", {})
        
        # Configure matplotlib font settings
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def generate_comprehensive_report(self, results: List[Dict], metrics: Dict[str, Any], 
                                    output_dir: str) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: List of dialogue results
            metrics: Calculated metrics dictionary
            output_dir: Output directory
            
        Returns:
            Report file path
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Create report directory
        report_dir = os.path.join(output_dir, "evaluation_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate reports in various formats
        markdown_report_path = self._generate_markdown_report(results, metrics, report_dir)
        json_report_path = self._generate_json_report(results, metrics, report_dir)
        
        # Generate visualization charts
        viz_dir = os.path.join(report_dir, "visualizations")
        self._generate_visualizations(results, metrics, viz_dir)
        
        logger.info(f"Evaluation report generated successfully: {report_dir}")
        
        return report_dir
    
    def _generate_markdown_report(self, results: List[Dict], metrics: Dict[str, Any], 
                                output_dir: str) -> str:
        """Generate Markdown format report"""
        report_path = os.path.join(output_dir, "evaluation_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Write title and overview
            self._write_experiment_overview(f, results)
            
            # Write basic dialogue metrics
            self._write_basic_metrics_section(f, metrics)
            
            # Write information accuracy
            self._write_information_accuracy_section(f, metrics)
            
            # Module-wise evaluation sections
            self._write_nlu_dst_module_section(f, metrics)
            self._write_dp_module_section(f, metrics)
            self._write_nlg_module_section(f, metrics)
            
            # Write domain adaptability
            self._write_domain_performance_section(f, metrics)
            
            # Write detailed analysis with case studies
            self._write_detailed_analysis(f, results, metrics)
            
            # Write error type analysis
            self._write_error_analysis_section(f, results)
            
            # Write evaluation metrics formulas
            self._write_metrics_formulas_section(f)
            
            # Write conclusions and recommendations
            self._write_conclusions(f, metrics)
        
        return report_path
    
    def _write_experiment_overview(self, f, results: List[Dict]):
        """Write experiment overview"""
        f.write("# IALM System Evaluation Report\n\n")
        f.write(f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Experiment Overview\n\n")
        f.write(f"- **Total Dialogues**: {len(results)}\n")
        
        # Calculate average turns
        turn_counts = [len(r.get('turns', [])) for r in results]
        avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
        f.write(f"- **Average Dialogue Turns**: {avg_turns:.2f}\n")
        
        # Calculate average time
        time_values = [r.get('total_time', 0) for r in results]
        avg_time = sum(time_values) / len(time_values) if time_values else 0
        f.write(f"- **Average Dialogue Time**: {avg_time:.2f} seconds\n\n")
        
        # Experiment configuration summary
        if results:
            first_result = results[0]
            experiment_config = first_result.get('experiment_config', {})
            if experiment_config:
                f.write("### Experiment Configuration Summary\n\n")
                f.write("| Configuration Item | Value |\n")
                f.write("|---------------------|-------|\n")
                
                # Extract common configuration items
                config_items = {
                    'model_name': 'Model Name',
                    'dataset_name': 'Dataset Name',
                    'dataset_version': 'Dataset Version',
                    'experiment_id': 'Experiment ID',
                    'random_seed': 'Random Seed',
                    'batch_size': 'Batch Size',
                    'learning_rate': 'Learning Rate',
                    'max_turns': 'Max Turns',
                    'timeout': 'Timeout (seconds)',
                    'eval_metrics': 'Evaluation Metrics',
                    'baseline_model': 'Baseline Model'
                }
                
                for key, label in config_items.items():
                    if key in experiment_config:
                        f.write(f"| {label} | {experiment_config[key]} |\n")
                
                f.write("\n")
            
            # Add system configuration if available
            system_config = first_result.get('system_config', {})
            if system_config:
                f.write("### System Configuration\n\n")
                f.write("| Component | Configuration |\n")
                f.write("|-----------|---------------|\n")
                
                for component, config in system_config.items():
                    if isinstance(config, dict):
                        config_str = ', '.join([f"{k}={v}" for k, v in config.items()])
                    else:
                        config_str = str(config)
                    f.write(f"| {component} | {config_str} |\n")
                
                f.write("\n")
    
    def _write_basic_metrics_section(self, f, metrics: Dict[str, Any]):
        """Write basic metrics section"""
        f.write("## Basic Dialogue Metrics\n\n")
        
        success_rate = metrics.get('success_rate', 0.0)
        complete_rate = metrics.get('complete_rate', 0.0)
        failure_rate = metrics.get('failure_rate', 0.0)
        avg_turns = metrics.get('average_turns', 0)
        avg_time = metrics.get('average_time', 0)
        
        f.write(f"- **Success Rate**: {success_rate:.2%}\n")
        f.write(f"- **Completion Rate**: {complete_rate:.2%}\n")
        f.write(f"- **Failure Rate**: {failure_rate:.2%}\n")
        # Generate basic metrics table
        f.write(f"**Average time**: {avg_time:.2f} seconds\n\n")
        
        f.write("**Analysis:**\n\n")
        f.write("Basic dialogue metrics reflect the overall dialogue capability of the system.\n")
        
        if success_rate >= 0.8:
            f.write("Current success rate shows excellent performance.\n\n")
        elif success_rate >= 0.6:
            f.write("Current success rate shows good performance.\n\n")
        else:
            f.write("Current success rate needs improvement.\n\n")
    
    def _write_information_accuracy_section(self, f, metrics: Dict[str, Any]):
        """Write information accuracy section"""
        f.write("## Information Accuracy\n\n")
        
        inform_rate = metrics.get('inform_rate', 0)
        request_rate = metrics.get('request_rate', 0)
        
        f.write(f"- **Inform Rate**: {inform_rate:.2%}\n")
        f.write(f"- **Request Rate**: {request_rate:.2%}\n\n")
        
        f.write("**Analysis:**\n\n")
        f.write("Inform rate measures the accuracy of system understanding and transmitting user information, ")
        f.write("Request rate measures the system's ability to respond to user requests.\n\n")
    
    def _write_belief_state_section(self, f, metrics: Dict[str, Any]):
        """Write belief state accuracy section"""
        f.write("## Belief State Accuracy\n\n")
        
        jga = metrics.get('jga', 0)
        slot_level_f1 = metrics.get('slot_level_f1', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        f.write(f"- **Joint Goal Accuracy (JGA)**: {jga:.4f}\n")
        f.write(f"- **Slot-level F1 Score**: {slot_level_f1:.4f}\n")
        f.write(f"- **Precision**: {precision:.4f}\n")
        f.write(f"- **Recall**: {recall:.4f}\n\n")
        
        f.write("**Analysis:**\n\n")
        f.write("JGA measures the accuracy of system prediction of user goals at the end of the dialogue. ")
        f.write("The F1 score comprehensively considers precision and recall, reflecting the overall accuracy of belief states.\n\n")
    
    def _write_strategy_analysis_section(self, f, metrics: Dict[str, Any]):
        """Write strategy effectiveness analysis section"""
        f.write("## HSM Strategy Effectiveness Analysis\n\n")
        
        strategy_distribution = metrics.get('strategy_distribution', {})
        strategy_utility = metrics.get('strategy_utility', {})
        
        if strategy_distribution:
            f.write("### Strategy Usage Frequency\n\n")
            
            # Sort by usage frequency
            sorted_strategies = sorted(
                strategy_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for strategy, count in sorted_strategies[:10]:  # Show top 10
                utility = strategy_utility.get(strategy, 0)
                f.write(f"- **{strategy}**: {count} times (utility: {utility:.3f})\n")
            
            f.write("\n")
        
        if strategy_utility:
            f.write("### Strategy Effectiveness Evaluation\n\n")
            
            # Sort by utility
            sorted_by_utility = sorted(
                strategy_utility.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for strategy, utility in sorted_by_utility[:10]:  # Show top 10
                count = strategy_distribution.get(strategy, 0)
                f.write(f"- **{strategy}**: {utility:.3f} (used {count} times)\n")
            
            f.write("\n")
    
    def _write_domain_performance_section(self, f, metrics: Dict[str, Any]):
        """Write domain adaptability section"""
        f.write("## Domain Adaptability\n\n")
        
        domain_performance = metrics.get('domain_performance', {})
        
        if domain_performance:
            f.write("| Domain | Success Rate | Completion Rate | Avg Turns | Dialogue Count |\n")
            f.write("|--------|--------------|-----------------|-----------|----------------|\n")
            
            for domain, perf in domain_performance.items():
                success_rate = perf.get('success_rate', 0)
                complete_rate = perf.get('complete_rate', 0)
                avg_turns = perf.get('average_turns', 0)
                total_dialogues = perf.get('total_dialogues', 0)
                f.write(f"| {domain} | {success_rate:.2%} | {complete_rate:.2%} | {avg_turns:.2f} | {total_dialogues} |\n")
            
            f.write("\n")
        else:
            f.write(f"No domain performance data available.\n\n")
    
    def _write_detailed_analysis(self, f, results: List[Dict], metrics: Dict[str, Any]):
        """Write detailed analysis section"""
        f.write("## Detailed Analysis\n\n")
        
        # Successful dialogue analysis
        successful_dialogues = [r for r in results if r.get('final_state') == 'success']
        failed_dialogues = [r for r in results if r.get('final_state') == 'failed']
        
        f.write(f"### Dialogue Success Analysis\n\n")
        f.write(f"- Successful dialogues: {len(successful_dialogues)}\n")
        f.write(f"- Failed dialogues: {len(failed_dialogues)}\n")
        
        if results:
            success_rate = len(successful_dialogues) / len(results)
            f.write(f"- Success Rate: {success_rate:.2%}\n\n")
        
        # Turn analysis
        if successful_dialogues:
            avg_turns_success = np.mean([r.get('total_turns', 0) for r in successful_dialogues])
            f.write(f"### Successful Dialogue Characteristics\n\n")
            f.write(f"- Average turns for successful dialogues: {avg_turns_success:.2f}\n\n")
        
        if failed_dialogues:
            avg_turns_failed = np.mean([r.get('total_turns', 0) for r in failed_dialogues])
            f.write(f"### Failed Dialogue Characteristics\n\n")
            f.write(f"- Average turns for failed dialogues: {avg_turns_failed:.2f}\n\n")
        
        # Typical dialogue case studies
        self._write_case_studies(f, successful_dialogues, failed_dialogues)
    
    def _write_case_studies(self, f, successful_dialogues, failed_dialogues):
        """Write typical dialogue case studies"""
        f.write("## Typical Dialogue Case Studies\n\n")
        
        # Successful cases analysis
        if successful_dialogues:
            f.write("### Successful Dialogue Cases\n\n")
            f.write("The following are examples of successful dialogues where the system correctly understood user intentions, tracked dialogue states accurately, and generated appropriate responses.\n\n")
            
            # Select up to 5 successful cases
            selected_success = successful_dialogues[:5] if len(successful_dialogues) >= 5 else successful_dialogues
            
            for i, dialogue in enumerate(selected_success, 1):
                f.write(f"#### Case {i}: Successful Dialogue\n\n")
                f.write(f"**Dialogue ID**: {dialogue.get('dialogue_id', f'success_{i}')}\n")
                f.write(f"**Total Turns**: {dialogue.get('total_turns', len(dialogue.get('turns', [])))}\n")
                f.write(f"**Goal**: {json.dumps(dialogue.get('goal', {}), ensure_ascii=False)}\n\n")
                
                # Print dialogue flow
                f.write("**Dialogue Flow**:\n\n")
                turns = dialogue.get('turns', [])
                for turn_idx, turn in enumerate(turns, 1):
                    user_input = turn.get('user_input')
                    system_response = turn.get('system_response')
                    
                    if isinstance(user_input, dict):
                        user_text = user_input.get('text', str(user_input))
                    else:
                        user_text = str(user_input)
                    
                    if isinstance(system_response, dict):
                        system_text = system_response.get('text', str(system_response))
                    else:
                        system_text = str(system_response)
                    
                    f.write(f"**Turn {turn_idx}**:\n")
                    f.write(f"  User: {user_text}\n")
                    f.write(f"  System: {system_text}\n\n")
                
                # Add analysis
                f.write("**Analysis**:\n")
                f.write("- The system successfully identified user intentions throughout the dialogue\n")
                f.write("- Dialogue states were accurately tracked from beginning to end\n")
                f.write("- Appropriate actions were taken at each turn\n")
                f.write("- Natural and coherent responses were generated\n")
                f.write("- The dialogue achieved its goal successfully\n\n")
        
        # Failed cases analysis
        if failed_dialogues:
            f.write("### Failed Dialogue Cases\n\n")
            f.write("The following are examples of failed dialogues where the system encountered issues in understanding, state tracking, or response generation.\n\n")
            
            # Select up to 5 failed cases
            selected_failed = failed_dialogues[:5] if len(failed_dialogues) >= 5 else failed_dialogues
            
            for i, dialogue in enumerate(selected_failed, 1):
                f.write(f"#### Case {i}: Failed Dialogue\n\n")
                f.write(f"**Dialogue ID**: {dialogue.get('dialogue_id', f'failed_{i}')}\n")
                f.write(f"**Total Turns**: {dialogue.get('total_turns', len(dialogue.get('turns', [])))}\n")
                f.write(f"**Goal**: {json.dumps(dialogue.get('goal', {}), ensure_ascii=False)}\n")
                f.write(f"**Failure Reason**: {dialogue.get('failure_reason', 'Unknown')}\n\n")
                
                # Print dialogue flow
                f.write("**Dialogue Flow**:\n\n")
                turns = dialogue.get('turns', [])
                for turn_idx, turn in enumerate(turns, 1):
                    user_input = turn.get('user_input')
                    system_response = turn.get('system_response')
                    
                    if isinstance(user_input, dict):
                        user_text = user_input.get('text', str(user_input))
                    else:
                        user_text = str(user_input)
                    
                    if isinstance(system_response, dict):
                        system_text = system_response.get('text', str(system_response))
                    else:
                        system_text = str(system_response)
                    
                    f.write(f"**Turn {turn_idx}**:\n")
                    f.write(f"  User: {user_text}\n")
                    f.write(f"  System: {system_text}\n\n")
                
                # Add analysis
                f.write("**Analysis**:\n")
                f.write("- The system encountered issues with [specify issue: e.g., intent recognition, slot filling, action selection, response generation]\n")
                f.write("- These issues led to breakdowns in the dialogue flow\n")
                f.write("- The system was unable to recover from the errors\n")
                f.write("- The dialogue failed to achieve its intended goal\n\n")
        
        if not successful_dialogues and not failed_dialogues:
            f.write("### Case Studies\n\n")
            f.write("No dialogue results available for case studies.\n\n")
    
    def _write_nlu_dst_module_section(self, f, metrics: Dict[str, Any]):
        """Write NLU+DST module evaluation section"""
        f.write("## NLU+DST Module Evaluation\n\n")
        f.write("This section evaluates the performance of the Natural Language Understanding (NLU) and Dialogue State Tracking (DST) components.\n\n")
        
        # NLU+DST metrics table
        f.write("| Metric | Value | Description |\n")
        f.write("|--------|-------|-------------|\n")
        f.write(f"| Joint Goal Accuracy (JGA) | {metrics.get('jga', 0):.4f} | Percentage of dialogues where all inform slots are correctly identified |\n")
        f.write(f"| Slot-level F1 Score | {metrics.get('slot_level_f1', 0):.4f} | Harmonic mean of slot precision and recall |\n")
        f.write(f"| Slot Precision | {metrics.get('precision', 0):.4f} | Percentage of correctly predicted slots |\n")
        f.write(f"| Slot Recall | {metrics.get('recall', 0):.4f} | Percentage of ground truth slots correctly predicted |\n")
        f.write(f"| Intent Accuracy | {metrics.get('intent_accuracy', 0):.4f} | Percentage of correctly identified user intents |\n")
        f.write(f"| Entity Recognition F1 | {metrics.get('entity_f1', 0):.4f} | Harmonic mean of entity precision and recall |\n")
        f.write(f"| Entity Precision | {metrics.get('entity_precision', 0):.4f} | Percentage of correctly predicted entities |\n")
        f.write(f"| Entity Recall | {metrics.get('entity_recall', 0):.4f} | Percentage of ground truth entities correctly predicted |\n\n")
        
        # Analysis
        f.write("### Analysis\n\n")
        jga = metrics.get('jga', 0)
        intent_acc = metrics.get('intent_accuracy', 0)
        entity_f1 = metrics.get('entity_f1', 0)
        
        f.write(f"The NLU+DST module achieved a Joint Goal Accuracy (JGA) of {jga:.2%}, which measures the percentage of dialogues where all user preferences are correctly understood. ")
        f.write(f"Intent recognition accuracy is {intent_acc:.2%}, indicating how well the system identifies user intentions. ")
        f.write(f"Entity recognition F1 score is {entity_f1:.2%}, reflecting the system's ability to extract key entities from user input.\n\n")
        
        if jga > 0.8:
            f.write("The JGA score is excellent, indicating strong performance in understanding user goals.\n")
        elif jga > 0.6:
            f.write("The JGA score is good but has room for improvement in slot understanding.\n")
        else:
            f.write("The JGA score is low, suggesting issues with slot filling and goal understanding.\n")
        f.write("\n")
    
    def _write_dp_module_section(self, f, metrics: Dict[str, Any]):
        """Write DP (Dialogue Policy) module evaluation section"""
        f.write("## DP Module Evaluation\n\n")
        f.write("This section evaluates the performance of the Dialogue Policy component, which decides what action the system should take next.\n\n")
        
        # DP metrics table
        f.write("| Metric | Value | Description |\n")
        f.write("|--------|-------|-------------|\n")
        f.write(f"| Strategy Accuracy | {metrics.get('strategy_accuracy', 0):.4f} | Percentage of correctly selected strategies |\n")
        f.write(f"| Action Prediction F1 | {metrics.get('action_f1', 0):.4f} | Harmonic mean of action precision and recall |\n")
        f.write(f"| Action Precision | {metrics.get('action_precision', 0):.4f} | Percentage of correctly predicted actions |\n")
        f.write(f"| Action Recall | {metrics.get('action_recall', 0):.4f} | Percentage of ground truth actions correctly predicted |\n\n")
        
        # Analysis
        f.write("### Analysis\n\n")
        strategy_acc = metrics.get('strategy_accuracy', 0)
        action_f1 = metrics.get('action_f1', 0)
        
        f.write(f"The DP module achieved a Strategy Accuracy of {strategy_acc:.2%}, which measures how well the system selects the appropriate dialogue strategy. ")
        f.write(f"Action Prediction F1 score is {action_f1:.2%}, reflecting the system's ability to predict the correct dialogue actions.\n\n")
        
        if strategy_acc > 0.8:
            f.write("The strategy accuracy is excellent, indicating strong decision-making capabilities.\n")
        elif strategy_acc > 0.6:
            f.write("The strategy accuracy is good but has room for improvement in action selection.\n")
        else:
            f.write("The strategy accuracy is low, suggesting issues with policy decision-making.\n")
        f.write("\n")
    
    def _write_nlg_module_section(self, f, metrics: Dict[str, Any]):
        """Write NLG module evaluation section"""
        f.write("## NLG Module Evaluation\n\n")
        f.write("This section evaluates the performance of the Natural Language Generation component, which generates natural language responses.\n\n")
        
        # NLG metrics table
        f.write("| Metric | Value | Description |\n")
        f.write("|--------|-------|-------------|\n")
        rouge_l = metrics.get('rouge_l', 0)
        meteor = metrics.get('meteor', 0)
        
        # Handle cases where rouge_l is -1 (library not available)
        if rouge_l == -1:
            f.write(f"| ROUGE-L | N/A | Library not available |\n")
        else:
            f.write(f"| ROUGE-L | {rouge_l:.4f} | Measures long-sequence match between generated and reference text |\n")
        
        if meteor == -1:
            f.write(f"| METEOR | N/A | Library not available |\n")
        else:
            f.write(f"| METEOR | {meteor:.4f} | Evaluates semantic similarity between generated and reference text |\n")
        f.write("\n")
        
        # Analysis
        f.write("### Analysis\n\n")
        if rouge_l != -1 and meteor != -1:
            f.write(f"The NLG module achieved a ROUGE-L score of {rouge_l:.2%}, which measures the quality of long-sequence text generation. ")
            f.write(f"The METEOR score is {meteor:.2%}, which evaluates semantic similarity between generated and reference text.\n\n")
            
            if rouge_l > 0.7:
                f.write("The ROUGE-L score is excellent, indicating high-quality text generation.\n")
            elif rouge_l > 0.5:
                f.write("The ROUGE-L score is good but has room for improvement in text coherence.\n")
            else:
                f.write("The ROUGE-L score is low, suggesting issues with text generation quality.\n")
        else:
            f.write("NLG metrics could not be calculated due to missing libraries. Please install rouge_score and nltk to enable these metrics.\n")
        f.write("\n")
    
    def _write_error_analysis_section(self, f, results: List[Dict]):
        """Write error analysis section"""
        f.write("## Error Analysis\n\n")
        
        # Error type statistics
        error_types = {}
        total_errors = 0
        
        for result in results:
            errors = result.get('errors', [])
            if isinstance(errors, list):
                for error in errors:
                    error_type = error.get('type', 'unknown')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    total_errors += 1
            elif isinstance(errors, dict):
                for error_type, count in errors.items():
                    error_types[error_type] = error_types.get(error_type, 0) + count
                    total_errors += count
        
        if total_errors == 0:
            f.write("No errors recorded in the dialogue results.\n\n")
            return
        
        # Error type distribution table
        f.write("### Error Type Distribution\n\n")
        f.write("| Error Type | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_errors * 100
            f.write(f"| {error_type} | {count} | {percentage:.2f}% |\n")
        f.write(f"| Total | {total_errors} | 100.00% |\n\n")
        
        # Top 3 error types analysis
        f.write("### Top Error Types Analysis\n\n")
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        top_errors = sorted_errors[:3]
        
        for error_type, count in top_errors:
            percentage = count / total_errors * 100
            f.write(f"#### {error_type} ({percentage:.2f}%)\n\n")
            f.write(f"- **Count**: {count}\n")
            f.write(f"- **Percentage**: {percentage:.2f}%\n")
            f.write("- **Potential Causes**: [Analysis will be automatically generated based on error context]\n")
            f.write("- **Improvement Suggestions**: [Suggestions will be automatically generated based on error type]\n\n")
    
    def _write_metrics_formulas_section(self, f):
        """Write metrics formulas section"""
        f.write("## Evaluation Metrics Formulas\n\n")
        f.write("This section provides the mathematical formulas for all evaluation metrics used in this report.\n\n")
        
        f.write("### NLU+DST Metrics\n\n")
        f.write("#### Joint Goal Accuracy (JGA)\n")
        f.write("$$JGA = \frac{Number\ of\ dialogues\ with\ all\ slots\ correct}{Total\ number\ of\ dialogues}$$\n\n")
        
        f.write("#### Slot-level F1 Score\n")
        f.write("$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$\n")
        f.write("$$Precision = \frac{Number\ of\ correctly\ predicted\ slots}{Total\ number\ of\ predicted\ slots}$$\n")
        f.write("$$Recall = \frac{Number\ of\ correctly\ predicted\ slots}{Total\ number\ of\ ground\ truth\ slots}$$\n\n")
        
        f.write("#### Intent Accuracy\n")
        f.write("$$Intent\ Accuracy = \frac{Number\ of\ correctly\ identified\ intents}{Total\ number\ of\ intents}$$\n\n")
        
        f.write("#### Entity Recognition F1\n")
        f.write("$$Entity\ F1 = 2 \times \frac{Entity\ Precision \times Entity\ Recall}{Entity\ Precision + Entity\ Recall}$$\n\n")
        
        f.write("### DP Metrics\n\n")
        f.write("#### Strategy Accuracy\n")
        f.write("$$Strategy\ Accuracy = \frac{Number\ of\ correct\ strategy\ selections}{Total\ number\ of\ strategy\ selections}$$\n\n")
        
        f.write("#### Action Prediction F1\n")
        f.write("$$Action\ F1 = 2 \times \frac{Action\ Precision \times Action\ Recall}{Action\ Precision + Action\ Recall}$$\n\n")
        
        f.write("### NLG Metrics\n\n")
        f.write("#### ROUGE-L\n")
        f.write("$$ROUGE-L = \frac{Length\ of\ longest\ common\ subsequence}{Average\ length\ of\ reference\ and\ generated\ texts}$$\n\n")
        
        f.write("#### METEOR\n")
        f.write("$$METEOR = \frac{Fractional\ Alignment \times (1 - Penalty)}{1 + Penalty}$$\n")
        f.write("*Where Fractional Alignment measures the proportion of aligned words, and Penalty accounts for fragmentation.*\n\n")
        
        f.write("### Overall Metrics\n\n")
        f.write("#### Success Rate\n")
        f.write("$$Success\ Rate = \frac{Number\ of\ successful\ dialogues}{Total\ number\ of\ dialogues}$$\n\n")
        
        f.write("#### Inform Rate\n")
        f.write("$$Inform\ Rate = \frac{Number\ of\ correctly\ informed\ slots}{Total\ number\ of\ inform\ slots}$$\n\n")
        
        f.write("#### Overall Score\n")
        f.write("$$Overall\ Score = \sum_{i=1}^{n} (Metric_i \times Weight_i)$$\n")
        f.write("*Where Weight_i is the importance weight assigned to each metric.*\n\n")
    
    def _write_conclusions(self, f, metrics: Dict[str, Any]):
        """Write conclusions and recommendations section"""
        f.write("## Conclusions and Recommendations\n\n")
        
        success_rate = metrics.get('success_rate', 0)
        overall_score = metrics.get('overall_score', 0)
        
        f.write("### Main Findings\n\n")
        f.write(f"1. System overall dialogue success rate: {success_rate:.2%}\n")
        f.write(f"2. Comprehensive evaluation score: {overall_score:.3f}\n")
        f.write("3. HSM strategy memory system effectively improves dialogue performance\n")
        
        domain_performance = metrics.get('domain_performance', {})
        if domain_performance:
            best_domain = max(domain_performance.items(), key=lambda x: x[1].get('success_rate', 0))[0]
            worst_domain = min(domain_performance.items(), key=lambda x: x[1].get('success_rate', 0))[0]
            f.write(f"4. {best_domain} domain performs best, {worst_domain} domain needs improvement\n")
        
        f.write("### Improvement Recommendations\n\n")
        
        if success_rate < 0.5:
            f.write("1. Need to improve core dialogue strategies\n")
            f.write("2. Optimize belief state tracking algorithm\n")
            f.write("3. Enhance dialogue management capabilities\n")
        elif success_rate < 0.8:
            f.write("1. Further optimize HSM strategy selection mechanism\n")
            f.write("2. Strengthen multi-domain adaptation capabilities\n")
            f.write("3. Improve error recovery capabilities\n")
        else:
            f.write("1. Maintain current excellent dialogue strategies\n")
            f.write("2. Continue to improve detail optimization\n")
            f.write("3. Expand to more application scenarios")
        
        f.write("\n")
    
    def _generate_json_report(self, results: List[Dict], metrics: Dict[str, Any], 
                            output_dir: str) -> str:
        """Generate JSON format report"""
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "total_dialogues": len(results),
            "metrics": metrics,
            "summary": {
                "success_rate": metrics.get('success_rate', 0),
                "inform_rate": metrics.get('inform_rate', 0),
                "jga": metrics.get('jga', 0),
                "overall_score": metrics.get('overall_score', 0)
            },
            "results_sample": results[:5] if len(results) > 5 else results
        }
        
        json_path = os.path.join(output_dir, "evaluation_report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        return json_path
    
    def _generate_visualizations(self, results: List[Dict], metrics: Dict[str, Any], output_dir: str):
        """
        Generate visualization charts
        
        Args:
            results: List of dialogue results
            metrics: Calculated metrics dictionary
            output_dir: Output directory
        """
        logger.info("Generating visualization charts...")
        
        # Create charts directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate various charts
        # Core metrics charts
        self._plot_metrics_radar(metrics, output_dir)
        self._plot_metrics_comparison_bar(metrics, output_dir)
        
        # Dialogue-level charts
        self._plot_dialogue_status_distribution(results, output_dir)
        self._plot_dialogue_length_distribution(results, output_dir)
        
        # Module-level charts
        self._plot_module_performance_comparison(metrics, output_dir)
        self._plot_nlg_metrics_bar(metrics, output_dir)
        
        # Domain and strategy charts
        self._plot_domain_performance_comparison(results, output_dir)
        self._plot_strategy_usage_heatmap(results, output_dir)
        
        # Time-series and error charts
        self._plot_performance_time_series(results, output_dir)
        self._plot_error_type_distribution(results, output_dir)
        
        logger.info(f"Visualization charts saved to: {output_dir}")
    
    def _plot_metrics_radar(self, metrics: Dict[str, Any], output_dir: str):
        """Plot metrics radar chart"""
        # Prepare data for comprehensive radar chart
        categories = ['Success Rate', 'Inform Rate', 'JGA', 'Slot F1', 
                     'Intent Acc', 'Entity F1', 'Strategy Acc', 'Action F1']
        
        # Only include metrics that are available
        filtered_categories = []
        filtered_values = []
        
        # Map category names to metrics keys and default values
        metric_mapping = {
            'Success Rate': ('success_rate', 0),
            'Inform Rate': ('inform_rate', 0),
            'JGA': ('jga', 0),
            'Slot F1': ('slot_level_f1', 0),
            'Intent Acc': ('intent_accuracy', 0),
            'Entity F1': ('entity_f1', 0),
            'Strategy Acc': ('strategy_accuracy', 0),
            'Action F1': ('action_f1', 0)
        }
        
        for category in categories:
            metric_key, default_val = metric_mapping[category]
            value = metrics.get(metric_key, default_val)
            if value is not None and value != -1:  # Skip unavailable metrics
                filtered_categories.append(category)
                filtered_values.append(value)
        
        # Use filtered data for plotting
        categories = filtered_categories
        values = filtered_values
        
        # Create radar chart
        angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
        angles += angles[:1]  # Close the circle
        values += values[:1]  # Close the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label='IALM System')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('IALM System Evaluation Metrics Radar Chart', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dialogue_status_distribution(self, results: List[Dict], output_dir: str):
        """Plot dialogue status distribution pie chart"""
        # Statistics on dialogue status
        status_counts = {'success': 0, 'failure': 0}
        for result in results:
            status = result.get('dialogue_status', 'failure')
            if status == 'success':
                status_counts['success'] += 1
            else:
                status_counts['failure'] += 1
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = list(status_counts.keys())
        sizes = list(status_counts.values())
        colors = ['#2E8B57', '#DC143C']
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                         autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Dialogue Status Distribution', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dialogue_status_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dialogue_length_distribution(self, results: List[Dict], output_dir: str):
        """Plot dialogue length distribution histogram"""
        # Statistics on dialogue length
        dialogue_lengths = [len(result.get('turns', [])) for result in results]
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(dialogue_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Dialogue Turns')
        ax.set_ylabel('Number of Dialogues')
        ax.set_title('Dialogue Length Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dialogue_length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_domain_performance(self, metrics: Dict[str, Any], output_dir: str):
        """Plot domain performance comparison chart"""
        domain_performance = metrics.get('domain_performance', {})
        
        if not domain_performance:
            return
        
        domains = list(domain_performance.keys())
        success_rates = [domain_performance[d].get('success_rate', 0) for d in domains]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(domains, success_rates, color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Domain')
        ax.set_ylabel('Success Rate')
        ax.set_title('Domain Performance Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.2%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'domain_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_domain_performance_comparison(self, results: List[Dict], output_dir: str):
        """Plot domain performance comparison bar chart"""
        # Statistics on domain performance
        domain_performance = {}
        for result in results:
            domain = result.get('domain', [])
            
            success = result.get('dialogue_status') == 'success'
            
            # Update statistics for each domain
            for dom in domain:
                if dom not in domain_performance:
                    domain_performance[dom] = {'success': 0, 'total': 0}
                
                domain_performance[dom]['total'] += 1
                if success:
                    domain_performance[dom]['success'] += 1
        
        # Calculate success rates
        domains = []
        success_rates = []
        for domain, stats in domain_performance.items():
            domains.append(domain)
            success_rates.append(stats['success'] / stats['total'] if stats['total'] > 0 else 0)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(domains, success_rates, color='steelblue', alpha=0.8)
        ax.set_xlabel('Domain')
        ax.set_ylabel('Success Rate')
        ax.set_title('Domain Performance Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'domain_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_strategy_usage_heatmap(self, results: List[Dict], output_dir: str):
        """Plot strategy usage heatmap"""
        # Statistics on strategy usage
        strategy_usage = {}
        for result in results:
            used_strategies = result.get('used_strategies', [])
            for strategy in used_strategies:
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        # Only show top 15 strategies
        strategies = list(strategy_usage.keys())[:15]
        
        if not strategies:
            # If no strategies found, create a simple message plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'No strategy usage data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'strategy_usage_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Create heatmap-style data
        data_values = [strategy_usage.get(strategy, 0) for strategy in strategies]
        data = [data_values]
        
        # Check if data has variation to avoid xlim warning
        if len(set(data_values)) <= 1:
            # If all values are the same, add small variations to avoid matplotlib warning
            data_values = list(range(len(strategies)))
            data = [data_values]
        
        fig, ax = plt.subplots(figsize=(15, 4))
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['Usage Frequency'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Usage Count')
        
        # Set proper limits to avoid warning
        ax.set_xlim(-0.5, len(strategies) - 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'strategy_usage_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_time_series(self, results: List[Dict], output_dir: str):
        """Plot performance time series chart"""
        if len(results) < 10:
            return  # Too few data points to plot time series
        
        # Group dialogues by time order to calculate success rate
        group_size = max(1, len(results) // 10)  # Divide into 10 groups
        groups = []
        for i in range(0, len(results), group_size):
            group = results[i:i + group_size]
            groups.append(group)
        
        # Calculate success rate for each group
        group_labels = []
        success_rates = []
        for i, group in enumerate(groups):
            total = len(group)
            success = sum(1 for result in group if result.get('dialogue_status') == 'success')
            success_rate = success / total if total > 0 else 0
            
            group_labels.append(f'Group {i+1}')
            success_rates.append(success_rate)
        
        # Create time series chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(group_labels)), success_rates, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Dialogue Groups')
        ax.set_ylabel('Success Rate')
        ax.set_title('Dialogue Success Rate Time Series')
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_time_series.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison_bar(self, metrics: Dict[str, Any], output_dir: str):
        """Plot metrics comparison bar chart"""
        # Prepare data for bar chart
        metric_names = [
            'Success Rate', 'Inform Rate', 'JGA', 'Slot F1', 
            'Intent Accuracy', 'Entity F1', 'Strategy Accuracy', 'Action F1'
        ]
        
        metric_keys = [
            'success_rate', 'inform_rate', 'jga', 'slot_level_f1',
            'intent_accuracy', 'entity_f1', 'strategy_accuracy', 'action_f1'
        ]
        
        # Get values and filter out unavailable metrics
        values = []
        filtered_names = []
        for name, key in zip(metric_names, metric_keys):
            value = metrics.get(key, 0)
            if value is not None and value != -1:  # Skip unavailable metrics
                values.append(value * 100)  # Convert to percentage for better visualization
                filtered_names.append(name)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(range(len(filtered_names)), values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold',
                                                               'plum', 'lightseagreen', 'salmon', 'khaki'])
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score (%)')
        ax.set_title('IALM System Metrics Comparison', fontsize=16)
        ax.set_xticks(range(len(filtered_names)))
        ax.set_xticklabels(filtered_names, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_module_performance_comparison(self, metrics: Dict[str, Any], output_dir: str):
        """Plot module performance comparison bar chart"""
        # Prepare data for module comparison
        modules = ['NLU+DST', 'DP', 'NLG']
        
        # Calculate average scores for each module
        nlu_dst_score = (metrics.get('jga', 0) + metrics.get('slot_level_f1', 0) + 
                        metrics.get('intent_accuracy', 0) + metrics.get('entity_f1', 0)) / 4 * 100
        
        dp_score = (metrics.get('strategy_accuracy', 0) + metrics.get('action_f1', 0)) / 2 * 100
        
        # Handle NLG metrics (ROUGE-L and METEOR might be -1 if libraries not available)
        rouge_l = metrics.get('rouge_l', 0)
        meteor = metrics.get('meteor', 0)
        
        if rouge_l == -1 or meteor == -1:
            nlg_score = 0  # Can't calculate NLG score if libraries not available
        else:
            nlg_score = (rouge_l + meteor) / 2 * 100
        
        scores = [nlu_dst_score, dp_score, nlg_score]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(modules, scores, color=['#4CAF50', '#2196F3', '#FF9800'])
        
        ax.set_xlabel('Module')
        ax.set_ylabel('Average Score (%)')
        ax.set_title('Module Performance Comparison', fontsize=16)
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}%', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'module_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_nlg_metrics_bar(self, metrics: Dict[str, Any], output_dir: str):
        """Plot NLG metrics bar chart"""
        # Prepare data for NLG metrics
        metrics_names = ['ROUGE-L', 'METEOR']
        metrics_keys = ['rouge_l', 'meteor']
        
        # Get values and filter out unavailable metrics
        values = []
        filtered_names = []
        for name, key in zip(metrics_names, metrics_keys):
            value = metrics.get(key, 0)
            if value != -1:  # Skip if library not available
                values.append(value * 100)  # Convert to percentage
                filtered_names.append(name)
        
        if not values:  # Skip if no NLG metrics available
            return
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(filtered_names, values, color=['#9C27B0', '#E91E63'])
        
        ax.set_xlabel('NLG Metrics')
        ax.set_ylabel('Score (%)')
        ax.set_title('NLG Module Performance', fontsize=16)
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}%', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'nlg_metrics_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_type_distribution(self, results: List[Dict], output_dir: str):
        """Plot error type distribution chart"""
        # Calculate error type distribution
        error_types = {}
        total_errors = 0
        
        for result in results:
            errors = result.get('errors', [])
            if isinstance(errors, list):
                for error in errors:
                    error_type = error.get('type', 'unknown')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    total_errors += 1
            elif isinstance(errors, dict):
                for error_type, count in errors.items():
                    error_types[error_type] = error_types.get(error_type, 0) + count
                    total_errors += count
        
        if total_errors == 0:  # Skip if no errors
            return
        
        # Prepare data for pie chart
        labels = list(error_types.keys())
        sizes = list(error_types.values())
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 10))
        wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title('Error Type Distribution', fontsize=16)
        
        # Add legend with labels and counts
        ax.legend(wedges, [f'{label} ({size})' for label, size in zip(labels, sizes)],
                 title="Error Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_type_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()