from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import logging
import json
import faiss
from sentence_transformers import SentenceTransformer
from ..utils.llm_client import LLMClient
from .esb_data import ESBData, ESBStrategy, ESBMetadata, StrategyMetadata
from .esb_utils import ESBUtils

logger = logging.getLogger(__name__)

class ESBEvolver:
    """ESB演进器，包含检索和演进功能"""
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """初始化ESB演进器"""
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.esb_data = None
        self.llm_client = None
        self.esb_datafile = self._get_esb_datafile_path()
        self._init_llm_client()
        self._load_esb_from_config()
    
    def _normalize_domain(self, domain: str) -> str:
        """
        将SGD的domain（如Services_1）转换为标准格式：去掉数字，字母转为小写
        
        Args:
            domain: 原始domain字符串
            
        Returns:
            标准化后的domain字符串
        """
        # 去掉下划线和数字
        normalized = ''.join([c for c in domain if not (c.isdigit() or c == '_')])
        # 转为小写
        normalized = normalized.lower()
        return normalized
    
    def _normalize_domains(self, domains: List[str]) -> List[str]:
        """
        标准化domain列表
        
        Args:
            domains: 原始domain列表
            
        Returns:
            标准化后的domain列表
        """
        return [self._normalize_domain(domain) for domain in domains]
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_esb_datafile_path(self) -> str:
        """获取ESB数据文件路径"""
        return self.config.get('memory', {}).get('esb', {}).get('esb_datafile', 'data/esb.json')
    
    def _load_esb_from_config(self):
        """从配置文件中加载ESB数据"""
        try:
            self.load_esb(self.esb_datafile)
            print(f"Successfully loaded ESB data from {self.esb_datafile}")
        except FileNotFoundError:
            print(f"ESB data file not found at {self.esb_datafile}, starting with empty ESB data")
            # 创建空的ESBData对象
            self.esb_data = ESBData(
                strategies=[],
                metadata=ESBMetadata()
            )
        except Exception as e:
            print(f"Error loading ESB data from {self.esb_datafile}: {e}")
            # 创建空的ESBData对象
            self.esb_data = ESBData(
                strategies=[],
                metadata=ESBMetadata()
            )
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            self.llm_client = LLMClient(
                provider="vllmLocal",
                config_path=self.config_path
            )
        except Exception as e:
            print(f"Error initializing LLM client: {e}")
            self.llm_client = None
    
    def load_esb(self, file_path: str) -> None:
        """加载ESB数据"""
        self.esb_data = ESBUtils.load_esb_data(file_path)
    
    def save_esb(self, file_path: str = None) -> None:
        """保存ESB数据"""
        if self.esb_data is None:
            raise ValueError("ESB data is not loaded")
        
        # 如果没有指定file_path，使用配置文件中的路径
        if file_path is None:
            file_path = self.esb_datafile
        
        # 更新metadata
        self.update_metadata()
        
        ESBUtils.save_esb_data(self.esb_data, file_path)
        print(f"Successfully saved ESB data to {file_path}")
    
    def save_esb_to_config(self) -> None:
        """保存ESB数据到配置文件指定的路径"""
        self.save_esb()
    
    def update_metadata(self) -> None:
        """更新ESB数据metadata，直接统计"""
        if self.esb_data is None:
            raise ValueError("ESB data is not loaded")
        
        # 使用ESBUtils中的方法来更新metadata
        self.esb_data = ESBUtils.update_esb_metadata(self.esb_data)
    
    # ========== 检索功能 ==========
    def recall_strategies(self, query_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """策略检索功能"""
        if self.esb_data is None:
            raise ValueError("ESB data is not loaded")
        
        # 获取查询信息
        agent_type = query_info.get('agent_type', '')
        raw_domains = query_info.get('domains', [])
        # 标准化domain
        domains = self._normalize_domains(raw_domains)
        top_k = query_info.get('top_k', 1)  # 只返回一个策略
        temperature = query_info.get('temperature', 0.5)
        
        # 过滤候选策略
        candidate_strategies = []
        for strategy in self.esb_data.strategies:
            # 匹配agent类型
            if agent_type and strategy.agent_type != agent_type:
                continue
            
            # 匹配domain - 策略的domain也需要标准化
            strategy_domains = self._normalize_domains(strategy.domains)
            if set(domains) != set(strategy_domains):
                continue
            
            candidate_strategies.append(strategy)
        
        if not candidate_strategies:
            # 触发冷启动生成策略，使用标准化后的domain
            self.cold_start(domains, agent_type)
            # 重新加载策略
            candidate_strategies = []
            for strategy in self.esb_data.strategies:
                # 匹配agent类型
                if agent_type and strategy.agent_type != agent_type:
                    continue
                
                # 匹配domain - 策略的domain也需要标准化
                strategy_domains = self._normalize_domains(strategy.domains)
                if set(domains) != set(strategy_domains):
                    continue
                
                candidate_strategies.append(strategy)
        
        if not candidate_strategies:
            return []
        
        # 使用轮盘赌算法选择策略
        selected_strategies = self._boltzmann_selection(candidate_strategies, temperature)
        
        # 将ESBStrategy对象转换为字典格式
        selected_strategies_dict = []
        for strategy in selected_strategies[:top_k]:
            # 策略使用次数自增1
            strategy.metadata.used_count += 1
            
            # 转换为字典格式
            strategy_dict = {
                'id': strategy.id,
                'type': strategy.agent_type,
                'content': strategy.content,
                'domains': strategy.domains,
                'reason': strategy.reason,
                'metadata': {
                    'helpful_count': strategy.metadata.helpful_count,
                    'harmful_count': strategy.metadata.harmful_count,
                    'used_count': strategy.metadata.used_count,
                    'generation': strategy.metadata.generation,
                    'alive': strategy.metadata.alive
                }
            }
            selected_strategies_dict.append(strategy_dict)
        
        # 保存更新后的ESB数据
        self.save_esb()
        
        # 返回top-k策略
        return selected_strategies_dict
    
    def _calculate_fitness(self, strategy: ESBStrategy) -> float:
        """计算策略适应度"""
        # 适应度函数: Φ(π) = α·(H+π-H-π)/(N_usedπ+ε) + β·Norm(g(π))
        alpha = 0.7  # 适应度权重
        beta = 0.3   # 代际权重
        epsilon = 1e-8  # 防止除零
        
        # 获取策略统计数据
        helpful_count = strategy.metadata.helpful_count
        harmful_count = strategy.metadata.harmful_count
        used_count = strategy.metadata.used_count
        generation = strategy.metadata.generation
        
        # 计算帮助率
        if used_count > 0:
            helpful_rate = (helpful_count - harmful_count) / (used_count + epsilon)
        else:
            helpful_rate = 0.5  # 默认值
        
        # 归一化生成代数
        # 获取所有策略的最大和最小生成代数
        all_generations = [s.metadata.generation for s in self.esb_data.strategies]
        if not all_generations:
            norm_generation = 0
        else:
            g_min = min(all_generations)
            g_max = max(all_generations)
            if g_max == g_min:
                norm_generation = 0.5
            else:
                norm_generation = (generation - g_min) / (g_max - g_min)
        
        # 计算最终适应度
        fitness = alpha * helpful_rate + beta * norm_generation
        
        return fitness
    
    def _boltzmann_selection(self, strategies: List[ESBStrategy], temperature: float) -> List[ESBStrategy]:
        """玻尔兹曼选择算法"""
        if not strategies:
            return []
        
        # 计算所有策略的适应度
        fitness_scores = [self._calculate_fitness(strategy) for strategy in strategies]
        
        # 计算玻尔兹曼概率
        if temperature == 0:
            # 贪婪选择，选择适应度最高的策略
            sorted_indices = np.argsort(fitness_scores)[::-1]
            return [strategies[i] for i in sorted_indices]
        else:
            # 计算指数适应度
            exp_fitness = np.exp(np.array(fitness_scores) / temperature)
            sum_exp_fitness = np.sum(exp_fitness)
            
            if sum_exp_fitness == 0:
                # 所有适应度相同，随机选择
                return list(np.random.permutation(strategies))
            
            # 计算概率分布
            probabilities = exp_fitness / sum_exp_fitness
            
            # 基于概率分布选择策略
            selected_indices = np.random.choice(len(strategies), size=len(strategies), p=probabilities, replace=True)
            selected_strategies = [strategies[i] for i in selected_indices]
            
            return selected_strategies
    
    
    # ========== 演进功能 ==========
    def cold_start(self, domains: List[str], agent_type: str) -> None:
        agent_roles = {
            'dst': 'Natural Language Understanding and Dialogue State Tracking',
            'dp': 'Dialogue Policy',
            'nlg': 'Natural Language Generation',
            'user_sim': 'User Simulator'
        }
        
        agent_role = agent_roles.get(agent_type, agent_type)
        num = 10

        # 标准化domain
        normalized_domains = self._normalize_domains(domains)

        # 生成domain字符串
        if len(normalized_domains) == 1:
            domain_str = normalized_domains[0]
        else:
            domain_str = ' & '.join(normalized_domains)
        
        user_prompt = f"""
## Task:
Generate {num} comprehensive optimization strategies for a specific module in a pipeline-based Task-Oriented Dialogue System.

## Domain(s):
{domain_str}

## Target Module:
{agent_type}({agent_role})

## Goal:
The strategies should aim to enhance the overall performance of the TODS by increasing task completion rate and reducing the average number of dialogue turns (improving efficiency).

## Requirements for Strategies:
1. Each strategy must be a self-contained, actionable recommendation.
2. Describe the strategy concisely in 3 to 5 sentences.
3. Focus on specific techniques, architectural adjustments, or training methods relevant to the target module.
4. Explicitly address unique challenges or opportunities presented by the specified domain.
5. Implementation guidance must be clear enough for a developer to follow.
6. Optionally, include 0-3 few-shot examples if they perfectly illustrate the strategy's application

## Number of Strategies: {num}

## Output Format
Output MUST be a valid JSON array only, with no additional text, explanations, or markdown formatting.
[
  {{
    "reason": "A clear, one sentence explanation of the performance bottleneck or optimization opportunity this strategy addresses for the specified module and domain.",
    "content": "The core strategy description and implementation steps (3-5 sentences). May include examples."
  }},
  ...continue for all {num} strategies
]
"""

        messages = [
            {"role": "system", "content": "You are an expert specializing in designing optimization strategies for pipeline-based Task-Oriented Dialogue Systems. Your expertise lies in developing comprehensive strategies that enhance the performance of different modules. These strategies are designed to improve system accuracy, user experience, and task completion rates through systematic performance optimization techniques."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm_client.generate_with_chat_format(messages)
            strategy_content = self.llm_client.clean_response(response)
        except Exception as e:
            print(f"Error generating strategy: {e}")
            strategy_content = "[]"

        # 解析JSON格式的策略内容
        try:
            strategies_data = json.loads(strategy_content)
            
            # 初始化ESBData如果还没有初始化
            if self.esb_data is None:
                self.esb_data = ESBData(
                    strategies=[],
                    metadata=ESBMetadata()
                )
            
            # 为每个策略创建ESBStrategy对象并添加到ESBData中
            for i, strategy_dict in enumerate(strategies_data):
                strategy_id = f"{agent_type}-001-{i:03d}"
                
                new_strategy = ESBStrategy(
                    id=strategy_id,
                    domains=normalized_domains,
                    content=strategy_dict.get('content', ''),
                    agent_type=agent_type,
                    reason=strategy_dict.get('reason', 'Generated from LLM'),
                    metadata=StrategyMetadata(
                        helpful_count=0,
                        harmful_count=0,
                        used_count=0,
                        generation=1,
                        alive=True
                    )
                )
                
                # 添加到ESBData中
                self.esb_data.strategies.append(new_strategy)
            
            # 更新metadata
            self.update_metadata()
            
            # 自动保存到配置文件指定的路径
            self.save_esb_to_config()
            
            print(f"Successfully generated {len(strategies_data)} strategies for {agent_type}: {domain_str}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON strategy content: {e}")
            print(f"Content: {strategy_content}")
        except Exception as e:
            print(f"Error processing strategy content: {e}")
    
    def prune_strategies(self) -> None:
        """剪枝算法：按domains分组，保留每个domains和agent_type的适应度最高的10个策略"""
        logger = logging.getLogger(__name__)
        
        if self.esb_data is None:
            logger.error("ESB data is not loaded")
            raise ValueError("ESB data is not loaded")
        
        try:
            # 1. 按domains分组策略（相同数量和值的domains为同一组）
            domains_groups = {}
            for strategy in self.esb_data.strategies:
                if not strategy.metadata.alive:
                    continue

                # 使用标准化后的domain进行分组
                current_domains = strategy.domains
                normalized_domains = self._normalize_domains(current_domains)
                current_domains_set = set(normalized_domains)
                
                # 查找是否已有相同的domains分组
                found_group = False
                for existing_key in domains_groups:
                    if set(existing_key) == current_domains_set:
                        domains_groups[existing_key].append(strategy)
                        found_group = True
                        break
                
                # 如果没有找到匹配的分组，创建新分组
                if not found_group:
                    domains_groups[tuple(normalized_domains)] = [strategy]
            
            # 2. 处理每个domains组
            for domains_key, strategies in domains_groups.items():
                domains = list(domains_key)
                
                # 3. 按agent_type分组当前domains组的策略
                agent_strategies = {}
                for strategy in strategies:
                    agent_type = strategy.agent_type
                    if agent_type not in agent_strategies:
                        agent_strategies[agent_type] = []
                    agent_strategies[agent_type].append(strategy)
                
                # 4. 处理当前domains组的每个agent_type组
                for agent_type, agent_strategy_list in agent_strategies.items():
                    alive_count = len(agent_strategy_list)
                    
                    # 如果alive策略数量超过10
                    if alive_count > 10:
                        # i. 按适应度降序排序
                        sorted_strategies = sorted(agent_strategy_list, key=self._calculate_fitness, reverse=True)
                        
                        # ii. 保留前10个策略为alive，其余设为false
                        for i, strategy in enumerate(sorted_strategies):
                            if i >= 10:
                                strategy.metadata.alive = False
                        
                        logger.info(f"Pruned {agent_type} strategies in domains {domains}: kept top 10, deactivated {alive_count - 10}")
            
            # 5. 更新metadata并保存
            self.update_metadata()
            self.save_esb()
        
        except Exception as e:
            logger.error(f"Error during strategy pruning: {str(e)}", exc_info=True)
            raise
    
    def merge_strategies(self) -> None:
        """合并相似策略"""
        logger = logging.getLogger(__name__)
        
        if self.esb_data is None:
            logger.error("ESB data is not loaded")
            raise ValueError("ESB data is not loaded")
        
        try:
            # Step 1: Load SentenceTransformer model for embeddings
            # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            # model = SentenceTransformer('intfloat/e5-small-v2')
            # model = SentenceTransformer('nomic-ai/nomic-embed-text-v1')
            # model = SentenceTransformer('BAAI/bge-base-en-v1.5')

            
            # Step 2: 按domains分组策略（相同数量和值的domains为同一组）
            domains_groups = {}
            for strategy in self.esb_data.strategies:
                if not strategy.metadata.alive:
                    continue
                    
                # 使用标准化后的domain进行分组
                current_domains = strategy.domains
                normalized_domains = self._normalize_domains(current_domains)
                current_domains_set = set(normalized_domains)
                agent_type = strategy.agent_type
                
                # 查找是否已有相同的domains分组
                found_group = False
                for existing_key in domains_groups:
                    if set(existing_key) == current_domains_set:
                        # 在该分组内按agent_type进一步分组
                        if agent_type not in domains_groups[existing_key]:
                            domains_groups[existing_key][agent_type] = []
                        domains_groups[existing_key][agent_type].append(strategy)
                        found_group = True
                        break
                
                # 如果没有找到匹配的分组，创建新分组
                if not found_group:
                    domains_groups[tuple(normalized_domains)] = {}
                    domains_groups[tuple(normalized_domains)][agent_type] = [strategy]
            
            # Step 3: 处理每个domains组
            for domains_key, agent_strategies in domains_groups.items():
                domains = list(domains_key)
                domains_str = ' & '.join(domains)
                
                # Step 4: 处理每个agent_type in current domains group
                for agent_type, strategies in agent_strategies.items():
                    
                    if len(strategies) < 2:
                        continue
                    
                    # Step 5: Generate embeddings for strategy contents
                    try:
                        strategy_contents = [s.content for s in strategies]
                        embeddings = model.encode(strategy_contents, convert_to_numpy=True)
                        # Normalize embeddings for cosine similarity
                        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    except Exception as e:
                        logger.error(f"Failed to generate embeddings for {agent_type} in {domains_str}: {str(e)}")
                        continue
                    
                    # Step 6: Build FAISS index
                    try:
                        dimension = embeddings.shape[1]
                        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                        index.add(embeddings)
                    except Exception as e:
                        logger.error(f"Failed to build FAISS index for {agent_type} in {domains_str}: {str(e)}")
                        continue
                    
                    # Step 7: Find all similar strategies with similarity > XX%
                    try:
                        # Search for all similar pairs
                        D, I = index.search(embeddings, k=len(strategies))  # Search all strategies
                        
                        # Create a set to track unique similar strategies
                        similar_strategy_ids = set()
                        
                        for i in range(len(strategies)):
                            for j in range(len(strategies)):
                                if i >= j:
                                    continue
                                if D[i][j] > 0.80:
                                    similar_strategy_ids.add(strategies[i].id)
                                    similar_strategy_ids.add(strategies[j].id)
                                                
                        if len(similar_strategy_ids) < 2:
                            continue
                        
                        # Get all similar strategies
                        similar_strategies = [s for s in strategies if s.id in similar_strategy_ids]
                        similar_strategy_ids_list = [s.id for s in similar_strategies]
                        logger.info(f"Processing similar strategies: {similar_strategy_ids_list}")
                        
                    except Exception as e:
                        logger.error(f"Failed to find similar pairs for {agent_type} in {domains_str}: {str(e)}")
                        continue
                    
                    # Step 8: Generate merge prompt with domain information
                    try:
                        # Format strategies for prompt
                        strategies_text = "\n".join([
                            f"Strategy {i+1}:\nContent: {s.content}\nReason: {s.reason}\n" 
                            for i, s in enumerate(similar_strategies)
                        ])
                        
                        prompt = f"""
## Task:
Merge multiple semantically similar strategies into one comprehensive strategy for {agent_type} module for {domains_str} domain(s) in a Task-Oriented Dialogue System.

## Merging Guidelines:
1. Analyze the provided strategies to identify: 1) Common themes and techniques, 2) Complementary ideas, 3) Domain-specific nuances.
2. Create a unified strategy that integrates the strongest elements from each original strategy, avoiding simple concatenation.
3. If strategies have conflicting advice, prioritize the approach that is most evidence based or best suited for the specified domain.
4. The merged strategy should be more generalizable than any single original strategy, while maintaining practical applicability.
5. Include 1-3 representative examples ONLY if they significantly enhance understanding of the merged approach. Adapt examples to better illustrate the integrated strategy.

{strategies_text}

# Output Format
Output ONLY a valid JSON object with exactly the structure below. Do not include any additional text, explanations, or markdown formatting.
{{
    "content": "Merged strategy description here",
    "reason": "Summary of the merged strategy's purpose and value"
}}
"""
                        
                        logger.debug(f"Generated merge prompt for strategies: {similar_strategy_ids_list}")
                        
                    except Exception as e:
                        logger.error(f"Failed to generate merge prompt for strategies {similar_strategy_ids_list}: {str(e)}")
                        continue
                    
                    # Step 9: Call LLM to generate merged strategy
                    try:
                        messages = [
                            {"role": "system", "content": "You are an expert specializing in designing optimization strategies for pipeline-based Task-Oriented Dialogue Systems. Your expertise lies in developing comprehensive strategies that enhance the performance of different modules. These strategies are designed to improve system accuracy, user experience, and task completion rates through systematic performance optimization techniques."},
                            {"role": "user", "content": prompt}
                        ]
                        
                        response = self.llm_client.generate_with_chat_format(messages)
                        response_content = self.llm_client.clean_response(response)
                        merged_result = json.loads(response_content)
                        
                        merged_content = merged_result.get('content', '')
                        merged_reason = merged_result.get('reason', '')
                        
                        if not merged_content:
                            logger.warning(f"Empty merged content for strategies {similar_strategy_ids_list}")
                            continue
                        
                    except Exception as e:
                        logger.error(f"Failed to generate merged strategy for strategies {similar_strategy_ids_list}: {str(e)}")
                        continue
                    
                    # Step 10: Update original strategies to alive=False
                    try:
                        for strategy in similar_strategies:
                            strategy.metadata.alive = False
                        logger.info(f"Marked {len(similar_strategies)} strategies as inactive: {similar_strategy_ids_list}")
                    except Exception as e:
                        logger.error(f"Failed to update original strategies {similar_strategy_ids_list}: {str(e)}")
                        continue
                    
                    # Step 11: Calculate metrics for new strategy
                    try:
                        # Calculate average counts
                        avg_helpful = np.mean([s.metadata.helpful_count for s in similar_strategies])
                        avg_harmful = np.mean([s.metadata.harmful_count for s in similar_strategies])
                        avg_used = np.mean([s.metadata.used_count for s in similar_strategies])
                        
                        # Determine new generation
                        max_generation = max([s.metadata.generation for s in similar_strategies])
                        new_generation = max_generation + 1
                        
                        # Generate new strategy ID
                        prefix = f"{agent_type}-{new_generation:03d}-"
                        max_suffix = self._get_max_strategy_suffix(prefix)
                        new_suffix = max_suffix + 1
                        new_id = f"{prefix}{new_suffix:03d}"
                        
                        logger.info(f"Calculated metrics for new strategy: helpful={avg_helpful:.2f}, harmful={avg_harmful:.2f}, used={avg_used:.2f}, generation={new_generation}")
                        
                    except Exception as e:
                        logger.error(f"Failed to calculate metrics for new strategy with parents {similar_strategy_ids_list}: {str(e)}")
                        continue
                    
                    # Step 12: Create new merged strategy
                    try:
                        new_strategy = ESBStrategy(
                            id=new_id,
                            domains=domains,
                            content=merged_content,
                            agent_type=agent_type,
                            reason=merged_reason,
                            metadata=StrategyMetadata(
                                helpful_count=avg_helpful,
                                harmful_count=avg_harmful,
                                used_count=avg_used,
                                generation=new_generation,
                                parents=similar_strategy_ids_list,
                                alive=True
                            )
                        )
                        
                        self.esb_data.strategies.append(new_strategy)
                        logger.info(f"Created new merged strategy: {new_id} for {agent_type} - {domains_str}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create new merged strategy for parents {similar_strategy_ids_list}: {str(e)}")
                        continue
            
            # Step 13: Update metadata and save
            try:
                self.update_metadata()
                self.save_esb()
                logger.info("Updated metadata and saved ESB data")
            except Exception as e:
                logger.error(f"Failed to update metadata and save ESB data: {str(e)}")
                raise
        
        except Exception as e:
            logger.error(f"Error during strategy merging: {str(e)}", exc_info=True)
            raise
    
    def mutate_strategies(self, goal, domains, agent_type, strategies, turns, evolve_data, success):
        """变异策略"""
        if self.esb_data is None:
            raise ValueError("ESB data is not loaded")
        
        logger = logging.getLogger(__name__)
        
        try:
            # 标准化domain
            normalized_domains = self._normalize_domains(domains)
            
            # 生成domain字符串
            if len(normalized_domains) == 1:
                domain_str = normalized_domains[0]
            else:
                domain_str = ' & '.join(normalized_domains)

            dialog_result = "SUCCESS" if success else "FAILURE"

            # 将数组格式转换为按agent_type分组的字典格式
            strategies_by_type = {}
            for strategy in strategies:
                agent_type = strategy.get('type', '')
                if agent_type not in strategies_by_type:
                    strategies_by_type[agent_type] = []
                
                # 只取strategy的content和reason字段
                simplified_strategy = {
                    'content': strategy.get('content', ''),
                    'reason': strategy.get('reason', '')
                }
                strategies_by_type[agent_type].append(simplified_strategy)
            
            # 特定智能体类型进化
            # 定义各智能体类型的优化目标
            optimization_goals = {
                "dst": "Improve belief state accuracy and update precision",
                "dp": "Reduce dialogue turns, improve task completion efficiency",
                "nlg": "Optimize naturalness and coherence of generated content, ensure context adaptability",
                "user_sim": "Optimize relevance and context adaptability of questions, precisely target belief state gaps while maintaining dialogue fluency",
                "e2e": "Improve belief state accuracy and update precision; Reduce dialogue turns, improve task completion efficiency, and finally improve the dialogue success rate",
            }
            
            # 获取当前智能体类型的优化目标
            agent_goal = optimization_goals.get(agent_type, "Optimize strategy performance")
            
            # 生成针对该智能体的专属提示词
            prompt = f"""
## Task:
You are an expert in Task-Oriented Dialogue Systems optimization. Based on the provided dialogue data, analyze and optimize the strategy for the {agent_type} module.

## Context:
- Target Module: {agent_type}
- Agent Goal: {agent_goal}
- Domain: {domain_str}
- Dialogue Result: {dialog_result}

## INPUT DATA

### Dialog Goal
{json.dumps(goal, indent=2)}

### Dialogue History
{json.dumps(turns, indent=2)}

### Current Strategies
{json.dumps(strategies_by_type, indent=2)}

### Feedback Analysis
{json.dumps(evolve_data, indent=2)}

## TASK INSTRUCTIONS

### Step 1: Strategy Evaluation
Score the current strategy's effectiveness in this dialogue:
- 1 (Helpful): Strategy contributed positively to dialogue success or efficiency
- 0 (Neutral): Strategy had no clear positive or negative impact
- -1 (Harmful): Strategy directly contributed to dialogue failure or inefficiency

Consider these factors when scoring:
1. Module specific performance in this dialogue
2. Impact on overall task completion
3. Contribution to dialogue efficiency (turn reduction)

### Step 2: Gap Analysis
Identify specific gaps between the current strategy and optimal performance by analyzing:
1. Dialogue failures or inefficiencies in the history
2. Feedback insights and recommendations
3. Domain specific challenges that emerged

### Step 3: Strategy Optimization
Create an updated strategy that addresses the identified gaps while maintaining effective aspects of the current strategy. Ensure the updated strategy:
1. Addresses Specific Issues: Directly targets problems observed in the dialogue
2. Provides Actionable Guidance: Clear, implementable recommendations
3. Leverages Domain Knowledge: Incorporates {domain_str} specific best practices
4. Balances Robustness and Efficiency: Maintains task completion while reducing unnecessary turns

### Step 4: Reasoning
Provide clear rationale explaining: What specific improvements the updated strategy makes

## Output Format
Output ONLY the JSON object. Do not include any additional text, explanations, or markdown formatting outside the JSON.
{{
    "strategy": {{
        "agent_type": "{agent_type}",
        "content": "Updated strategy description (1-3 few-shot expamles if needed)",
        "reason": "Clear rationale for the update one sentence explaining what issues were addressed)",
        "score": 1|0|-1
    }}
}}
"""
                
            messages = [
                {"role": "system", "content": f"You are an expert in optimizing the {agent_type} module of task-oriented dialogue systems. Your expertise lies in developing strategies that enhance the performance of this specific module."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = self.llm_client.generate_with_chat_format(messages)
                response_content = self.llm_client.clean_response(response)
                result = json.loads(response_content)
                updated_strategy = result.get('strategy', {})
                
                # 处理更新后的策略
                self._update_strategies([updated_strategy], strategies)
                self.update_metadata()
                self.save_esb()
                
            except Exception as e:
                logger.error(f"Error in {agent_type} agent evolution: {str(e)}")

        except Exception as e:
            logger.error(f"Error during strategy mutation: {str(e)}", exc_info=True)
            raise
    
    def process_evolve(self, dialog_data):
        """处理对话策略进化"""
        logger = logging.getLogger(__name__)
        
        if self.esb_data is None:
            logger.error("ESB data is not loaded")
            raise ValueError("ESB data is not loaded")
        
        try:
            need_evolve = dialog_data.get('need_evolve', {})
            success = dialog_data.get('success', False)
            turns = dialog_data.get('turns', [])
            domains = dialog_data.get('domain', [])
            goal = dialog_data.get('goal', {})
            
            # 1. 执行变异操作
            if need_evolve or not success:
                # 从对话数据中提取各智能体类型对应的need_evolve取值及策略
                strategies = dialog_data.get('strategies', {})
                for agent_type, evolve_data in need_evolve.items():
                    if not evolve_data:
                        evolve_data = {}
                    agent_strategies = strategies.get(agent_type, [])
                    # 传递原始domains，在mutate_strategies中会进行标准化
                    self.mutate_strategies(goal, domains, agent_type, agent_strategies, turns, evolve_data, success)
            
            # 2. 执行合并操作
            self.merge_strategies()
            
            # 3. 执行剪枝操作
            self.prune_strategies()
        
        except Exception as e:
            logger.error(f"Error during dialogue strategy evolution: {str(e)}", exc_info=True)
            raise
    
    def _get_max_strategy_suffix(self, prefix: str) -> int:
        """获取指定前缀的策略ID中后3位数字的最大值
        
        Args:
            prefix: 策略ID前缀，格式为 "{agent_type}-{generation:03d}-"
            
        Returns:
            int: 后3位数字的最大值，如果没有找到匹配的策略ID则返回-1
        """
        if self.esb_data is None:
            raise ValueError("ESB data is not loaded")
        
        max_suffix = -1
        
        for strategy in self.esb_data.strategies:
            if strategy.id.startswith(prefix):
                # 提取后3位数字
                try:
                    suffix = strategy.id.split('-')[2]
                    suffix_num = int(suffix)
                    if suffix_num > max_suffix:
                        max_suffix = suffix_num
                except (IndexError, ValueError):
                    # 如果ID格式不符合预期，跳过
                    continue
        
        return max_suffix
    
    def _update_strategies(self, updated_strategies, original_strategies):
        """更新策略"""
        if self.esb_data is None:
            raise ValueError("ESB data is not loaded")
        
        logger = logging.getLogger(__name__)
        
        for updated_strategy in updated_strategies:
            agent_type = updated_strategy.get('agent_type', '')
            content = updated_strategy.get('content', '')
            reason = updated_strategy.get('reason', '')
            score = updated_strategy.get('score', 0)
            
            # 找到对应的原策略
            original_strategy = None
            for os in original_strategies:
                if os.get('type') == agent_type:
                    original_strategy = os
                    break
            
            if original_strategy:
                # 更新原策略的helpful或harmful分数
                original_strategy_id = original_strategy.get('id', '')
                for strategy in self.esb_data.strategies:
                    if strategy.id == original_strategy_id:
                        if score == 1:
                            strategy.metadata.helpful_count += 1
                        elif score == -1:
                            strategy.metadata.harmful_count += 1
                        break
                
                # 生成新策略
                # 从原策略ID中提取generation，在原基础上加1
                original_id_parts = original_strategy_id.split('-')
                if len(original_id_parts) >= 2:
                    generation = int(original_id_parts[1]) + 1
                    
                    # Generate new strategy ID
                    prefix = f"{agent_type}-{generation:03d}-"
                    max_suffix = self._get_max_strategy_suffix(prefix)
                    new_suffix = max_suffix + 1
                    new_strategy_id = f"{prefix}{new_suffix:03d}"
                    
                    # 获取原策略的domains并标准化
                    domains = original_strategy.get('domains', [])
                    normalized_domains = self._normalize_domains(domains)
                    
                    # 创建新策略
                    new_strategy = ESBStrategy(
                        id=new_strategy_id,
                        domains=normalized_domains,
                        content=content,
                        agent_type=agent_type,
                        reason=reason,
                        metadata=StrategyMetadata(
                            helpful_count=0,
                            harmful_count=0,
                            used_count=0,
                            generation=generation,
                            parents=[original_strategy_id],
                            alive=True
                        )
                    )
                    
                    # 加入到ESB中
                    self.esb_data.strategies.append(new_strategy)
                    logger.info(f"Created new strategy: {new_strategy_id} for {agent_type}")
