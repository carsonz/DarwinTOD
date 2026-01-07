import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from .esb_utils import ESBUtils
from .esb_evolver import ESBEvolver
from .esb_data import ESBData, ESBStrategy

# 配置变量
CONFIG = {
    'esb_data_path': 'data/esb.json',
    'output_dir': 'output/esb_analysis',
    'bert_model': 'BAAI/bge-small-en-v1.5',
    'dpi': 300,
    'random_state': 42
}

# 确保输出目录存在
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# 初始化模型
sbert_model = SentenceTransformer(CONFIG['bert_model'])


def calculate_entropy(text: str) -> float:
    """计算文本的信息熵
    
    Args:
        text: 输入文本
    
    Returns:
        信息熵值
    """
    from collections import Counter
    
    # 按空格分割文本为单词
    words = text.split()
    
    # 如果没有单词，返回0
    if not words:
        return 0.0
    
    # 计算单词频率
    freq_dict = Counter(words)
    total_words = len(words)
    
    # 计算信息熵
    entropy = 0.0
    for freq in freq_dict.values():
        prob = freq / total_words
        entropy -= prob * np.log2(prob)
    
    return entropy


def calculate_fitness(strategy: ESBStrategy, esb_data: ESBData) -> float:
    """计算策略适应度
    
    Args:
        strategy: ESBStrategy对象
        esb_data: ESBData对象
    
    Returns:
        适应度值
    """
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
    all_generations = [s.metadata.generation for s in esb_data.strategies]
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


def preprocess_data() -> Dict[str, Any]:
    """数据预处理模块
    
    Returns:
        预处理结果字典
    """
    print("开始数据预处理...")
    
    # 1. 读取ESB数据
    esb_data = ESBUtils.load_esb_data(CONFIG['esb_data_path'])
    
    # 2. 按agent_type和generation分组
    grouped_data = {}
    for strategy in esb_data.strategies:
        agent_type = strategy.agent_type
        generation = strategy.metadata.generation
        
        if agent_type not in grouped_data:
            grouped_data[agent_type] = {}
        if generation not in grouped_data[agent_type]:
            grouped_data[agent_type][generation] = []
        
        grouped_data[agent_type][generation].append(strategy)
    
    # 3. 计算各项指标
    processed_data = {
        'agent_types': list(grouped_data.keys()),
        'generations': {},
        'all_data': []
    }
    
    for agent_type, generations in grouped_data.items():
        processed_data['generations'][agent_type] = list(generations.keys())
        
        for generation, strategies in generations.items():
            for strategy in strategies:
                # 计算SBERT嵌入
                sbert_embedding = sbert_model.encode(strategy.content, convert_to_numpy=True).tolist()
                
                # 计算信息熵
                entropy = calculate_entropy(strategy.content)
                
                # 计算适应度
                fitness = calculate_fitness(strategy, esb_data)
                
                # 构建结果字典
                result = {
                    "agent_type": agent_type,
                    "meta_data": {
                        "generation": generation,
                        "ID": strategy.id,
                        "alive": strategy.metadata.alive,
                        "sbert": sbert_embedding,
                        "entropy": entropy,
                        "fitness": fitness,
                        "total": {
                            "entropy": entropy,
                            "kde": 0.0,  # 后续计算
                            "fitness": fitness
                        }
                    }
                }
                
                processed_data['all_data'].append(result)
    
    # 4. 计算核密度估计值
    from scipy.stats import gaussian_kde
    
    # 按agent_type和generation分组计算kde
    for agent_type, generations in grouped_data.items():
        for generation, strategies in generations.items():
            # 获取当前agent_type和generation的所有策略
            current_data = [d for d in processed_data['all_data'] 
                          if d['agent_type'] == agent_type 
                          and d['meta_data']['generation'] == generation]
            
            if len(current_data) < 2:
                continue
            
            # 提取fitness值
            fitness_values = np.array([d['meta_data']['fitness'] for d in current_data])
            
            # 计算核密度估计
            max_kde = 0.0
            
            # 检查是否所有值都相同
            if len(np.unique(fitness_values)) > 1:
                try:
                    kde = gaussian_kde(fitness_values)
                    
                    # 生成用于计算KDE的值范围
                    x = np.linspace(fitness_values.min(), fitness_values.max(), 100)
                    
                    # 计算KDE值
                    kde_values = kde(x)
                    
                    # 简化kde结果，取最大值作为代表
                    max_kde = np.max(kde_values) if len(kde_values) > 0 else 0.0
                except (np.linalg.LinAlgError, ValueError):
                    # 处理奇异矩阵或其他计算错误
                    max_kde = 0.0
            
            # 更新每个策略的kde值
            for d in current_data:
                d['meta_data']['total']['kde'] = max_kde
    
    # 5. 保存预处理结果
    output_path = os.path.join(CONFIG['output_dir'], 'preprocessed_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"数据预处理完成，结果保存到: {output_path}")
    return processed_data


def calculate_entropy_by_generation(processed_data: Dict[str, Any]) -> pd.DataFrame:
    """信息熵计算模块
    
    Args:
        processed_data: 预处理结果字典
    
    Returns:
        信息熵数据DataFrame
    """
    print("开始计算信息熵...")
    
    # 按agent_type和generation分组计算平均信息熵
    entropy_data = []
    for agent_type in processed_data['agent_types']:
        agent_generations = processed_data['generations'][agent_type]
        max_gen = max(agent_generations)
        
        # 构建当前agent_type的数据行
        agent_row = {'agent_type': agent_type}
        
        # 计算每个generation的平均信息熵
        for gen in range(1, max_gen + 1):
            # 获取当前generation的数据
            gen_data = [d for d in processed_data['all_data'] 
                       if d['agent_type'] == agent_type 
                       and d['meta_data']['generation'] == gen]
            
            if gen_data:
                avg_entropy = np.mean([d['meta_data']['entropy'] for d in gen_data])
                agent_row[f'generation_{gen}'] = avg_entropy
            else:
                agent_row[f'generation_{gen}'] = np.nan
        
        entropy_data.append(agent_row)
    
    # 转换为DataFrame
    df_entropy = pd.DataFrame(entropy_data)
    
    # 保存为CSV文件
    output_path = os.path.join(CONFIG['output_dir'], 'entropy_by_generation.csv')
    df_entropy.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"信息熵计算完成，结果保存到: {output_path}")
    return df_entropy


def prepare_tsne_data(processed_data: Dict[str, Any]) -> pd.DataFrame:
    """t-SNE数据准备模块
    
    Args:
        processed_data: 预处理结果字典
    
    Returns:
        t-SNE结果DataFrame
    """
    print("开始准备t-SNE数据...")
    
    # 1. 初始化t-SNE数据列表
    tsne_data = []
    
    # 2. 按agent_type分组处理
    for agent_type in processed_data['agent_types']:
        # 准备当前agent_type的数据
        agent_data = [d for d in processed_data['all_data'] if d['agent_type'] == agent_type]
        
        # 提取当前agent_type的所有generation信息
        agent_generations = [d['meta_data']['generation'] for d in agent_data]
        
        if not agent_generations:
            continue
        
        min_gen = 1
        max_gen = max(agent_generations)
        
        # 3. 提取当前agent_type的generation=1和所有alive的策略数据
        for d in agent_data:
            gen = d['meta_data']['generation']
            # 提取generation=1或所有alive的策略
            if gen == min_gen or d['meta_data']['alive']:
                tsne_data.append({
                    'agent_type': d['agent_type'],
                    'generation': gen,
                    'alive': d['meta_data']['alive'],
                    'sbert': d['meta_data']['sbert'],
                    'fitness': d['meta_data']['fitness']
                })
    
    # 4. 转换为DataFrame
    df_tsne = pd.DataFrame(tsne_data)
    
    if df_tsne.empty:
        print("没有可用的t-SNE数据")
        return df_tsne
    
    # 5. 准备SBERT向量矩阵
    sbert_vectors = np.array(df_tsne['sbert'].tolist())
    
    # 6. 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=CONFIG['random_state'], perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(sbert_vectors)
    
    # 7. 添加t-SNE结果到DataFrame
    df_tsne['tsne_x'] = tsne_results[:, 0]
    df_tsne['tsne_y'] = tsne_results[:, 1]
    
    # 8. 保存为CSV文件
    output_path = os.path.join(CONFIG['output_dir'], 'tsne_results.csv')
    df_tsne[['agent_type', 'generation', 'alive', 'tsne_x', 'tsne_y', 'fitness']].to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"t-SNE数据准备完成，结果保存到: {output_path}")
    return df_tsne


def plot_fitness_entropy_trend(processed_data: Dict[str, Any]) -> None:
    """图表绘制模块1（Fitness与Entropy趋势图）
    
    Args:
        processed_data: 预处理结果字典
    """
    print("开始绘制Fitness与Entropy趋势图...")
    
    # 1. 准备数据
    trend_data = []
    for d in processed_data['all_data']:
        trend_data.append({
            'agent_type': d['agent_type'],
            'generation': d['meta_data']['generation'],
            'entropy': d['meta_data']['entropy'],
            'fitness': d['meta_data']['fitness'],
            'sbert': d['meta_data']['sbert']
        })
    
    df_trend = pd.DataFrame(trend_data)
    
    # 2. 按agent_type分组绘制图表
    for agent_type in processed_data['agent_types']:
        # 获取当前agent_type的数据
        df_agent = df_trend[df_trend['agent_type'] == agent_type]
        
        # 计算每个generation的平均entropy和fitness
        df_agent_avg = df_agent.groupby('generation')[['entropy', 'fitness']].mean().reset_index()
        
        # 计算每个generation的策略两两余弦相似度的平均值
        cosine_similarities = []
        for gen in df_agent_avg['generation']:
            gen_data = df_agent[df_agent['generation'] == gen]
            if len(gen_data) < 2:
                cosine_similarities.append(np.nan)
            else:
                # 提取SBERT向量
                sbert_vectors = np.array(gen_data['sbert'].tolist())
                # 计算余弦相似度矩阵
                sim_matrix = cosine_similarity(sbert_vectors)
                # 获取上三角矩阵的值（不包括对角线）
                upper_tri = sim_matrix[np.triu_indices(len(sbert_vectors), k=1)]
                # 计算平均值
                avg_sim = np.mean(upper_tri) if len(upper_tri) > 0 else np.nan
                cosine_similarities.append(avg_sim)
        
        df_agent_avg['cosine_similarity'] = cosine_similarities
        
        # 3. 创建组合图表
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 左侧y轴：fitness柱状图
        ax1.bar(df_agent_avg['generation'], df_agent_avg['fitness'], color='skyblue', alpha=0.7, label='Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        
        # 右侧y轴1：entropy折线图
        ax2 = ax1.twinx()
        ax2.plot(df_agent_avg['generation'], df_agent_avg['entropy'], color='red', marker='o', 
                linewidth=2, markersize=6, label='Entropy')
        ax2.set_ylabel('Entropy', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 右侧y轴2：余弦相似度折线图
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(df_agent_avg['generation'], df_agent_avg['cosine_similarity'], 
                color='green', marker='s', linewidth=2, markersize=6, label='Avg Cosine Similarity')
        ax3.set_ylabel('Avg Cosine Similarity', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        
        # 添加标题和图例
        plt.title(f'Fitness, Entropy and Cosine Similarity Trend for {agent_type}')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        output_path = os.path.join(CONFIG['output_dir'], f'fitness_entropy_trend_{agent_type}.png')
        plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    print("Fitness与Entropy趋势图绘制完成")


def plot_tsne_visualization(df_tsne: pd.DataFrame) -> None:
    """图表绘制模块2（t-SNE可视化图）
    
    Args:
        df_tsne: t-SNE结果DataFrame
    """
    print("开始绘制t-SNE可视化图...")
    
    # 获取所有agent_type
    agent_types = df_tsne['agent_type'].unique()
    min_gen = 1
    
    # 为每个agent_type分别绘制图表
    for agent_type in agent_types:
        # 1. 绘制当前agent_type的generation=1的t-SNE图
        df_gen1 = df_tsne[(df_tsne['agent_type'] == agent_type) & (df_tsne['generation'] == min_gen)]
        
        if not df_gen1.empty:
            # 创建图表
            plt.figure(figsize=(12, 10))
            
            # 绘制散点图，颜色根据fitness值
            scatter = plt.scatter(df_gen1['tsne_x'], df_gen1['tsne_y'], 
                                c=df_gen1['fitness'], cmap='YlOrRd', 
                                s=60, alpha=0.8)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('Fitness')
            
            # 添加标题和坐标轴标签
            plt.title(f't-SNE Visualization of SBERT Embeddings (Agent: {agent_type}, Generation: {min_gen})')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            # 添加网格
            plt.grid(True, alpha=0.3)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(CONFIG['output_dir'], f'tsne_visualization_{agent_type}_generation_{min_gen}.png')
            plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
            plt.close()
        
        # 2. 绘制当前agent_type的所有alive策略的t-SNE图
        df_alive = df_tsne[(df_tsne['agent_type'] == agent_type) & (df_tsne['alive'] == True)]
        
        if not df_alive.empty:
            # 创建图表
            plt.figure(figsize=(12, 10))
            
            # 绘制散点图，颜色根据fitness值
            scatter = plt.scatter(df_alive['tsne_x'], df_alive['tsne_y'], 
                                c=df_alive['fitness'], cmap='YlOrRd', 
                                s=60, alpha=0.8)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('Fitness')
            
            # 添加标题和坐标轴标签
            plt.title(f't-SNE Visualization of SBERT Embeddings (Agent: {agent_type}, Alive Strategies)')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            # 添加网格
            plt.grid(True, alpha=0.3)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(CONFIG['output_dir'], f'tsne_visualization_{agent_type}_alive.png')
            plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
            plt.close()
    
    print("t-SNE可视化图绘制完成")


def plot_fitness_distribution(processed_data: Dict[str, Any]) -> None:
    """图表绘制模块3（适应度分布核密度图）
    
    Args:
        processed_data: 预处理结果字典
    """
    print("开始绘制适应度分布核密度图...")
    
    # 1. 准备数据
    fitness_data = []
    for d in processed_data['all_data']:
        fitness_data.append({
            'agent_type': d['agent_type'],
            'generation': d['meta_data']['generation'],
            'fitness': d['meta_data']['fitness']
        })
    
    df_fitness = pd.DataFrame(fitness_data)
    
    # 2. 按agent_type分组绘制
    for agent_type in processed_data['agent_types']:
        # 获取当前agent_type的数据
        df_agent = df_fitness[df_fitness['agent_type'] == agent_type]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 获取所有generations
        generations = sorted(df_agent['generation'].unique())
        
        # 绘制核密度图
        for gen in generations:
            # 获取当前generation的数据
            gen_fitness = df_agent[df_agent['generation'] == gen]['fitness'].dropna()
            
            if len(gen_fitness) > 0:
                # 检查数据集方差，如果方差为0则跳过KDE绘制
                if gen_fitness.var() > 0:
                    sns.kdeplot(gen_fitness, label=f'Generation {gen}', linewidth=2)
                    
                    # 计算并标注均值和中位数
                    mean_val = gen_fitness.mean()
                    median_val = gen_fitness.median()
                    plt.axvline(mean_val, color=plt.gca().lines[-1].get_color(), 
                              linestyle='--', alpha=0.7)
                    plt.axvline(median_val, color=plt.gca().lines[-1].get_color(), 
                              linestyle=':', alpha=0.7)
                else:
                    # 方差为0时，只绘制均值和中位数线，不绘制KDE
                    mean_val = gen_fitness.mean()
                    median_val = gen_fitness.median()
                    # 使用固定颜色
                    color = f'C{generations.index(gen)}'
                    plt.axvline(mean_val, color=color, linestyle='--', alpha=0.7, label=f'Generation {gen} (mean)')
                    plt.axvline(median_val, color=color, linestyle=':', alpha=0.7)
        
        # 添加标题和坐标轴标签
        plt.title(f'Fitness Distribution by Generation for {agent_type}')
        plt.xlabel('Fitness')
        plt.ylabel('Density')
        
        # 添加图例
        plt.legend()
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(CONFIG['output_dir'], f'fitness_distribution_{agent_type}.png')
        plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    print("适应度分布核密度图绘制完成")


def plot_sbert_clustering(processed_data: Dict[str, Any]) -> None:
    """图表绘制模块4（SBERT聚类分析图）
    
    Args:
        processed_data: 预处理结果字典
    """
    print("开始绘制SBERT聚类分析图...")
    
    # 1. 按agent_type分组处理
    for agent_type in processed_data['agent_types']:
        # 准备数据
        agent_data = [d for d in processed_data['all_data'] if d['agent_type'] == agent_type]
        
        # 提取当前agent_type的所有generation信息
        agent_generations = []
        for d in agent_data:
            agent_generations.append(d['meta_data']['generation'])
        
        if not agent_generations:
            continue
        
        min_gen = 1
        max_gen = max(agent_generations)
        
        # 处理generation=1的数据
        gen_data = [d for d in agent_data if d['meta_data']['generation'] == min_gen]
        
        if len(gen_data) >= 2:
            # 绘制第一代聚类分析图
            # 提取SBERT向量
            sbert_vectors = np.array([d['meta_data']['sbert'] for d in gen_data])
            
            # 3. 聚类分析
            # 尝试不同的聚类数量，选择最优的
            best_n_clusters = 2
            best_silhouette = -1
            
            for n_clusters in range(2, min(10, len(gen_data)) + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=CONFIG['random_state'])
                clusters = kmeans.fit_predict(sbert_vectors)
                
                # 计算轮廓系数
                silhouette = silhouette_score(sbert_vectors, clusters)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_n_clusters = n_clusters
            
            # 使用最优聚类数量
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=CONFIG['random_state'])
            clusters = kmeans.fit_predict(sbert_vectors)
            
            # 4. 计算类内距离
            cluster_distances = []
            for cluster_id in range(best_n_clusters):
                cluster_vectors = sbert_vectors[clusters == cluster_id]
                if len(cluster_vectors) > 1:
                    # 计算类内平均距离
                    centroid = cluster_vectors.mean(axis=0)
                    distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
                    avg_distance = np.mean(distances)
                    cluster_distances.append(avg_distance)
            
            avg_intra_cluster_distance = np.mean(cluster_distances) if cluster_distances else 0
            
            # 5. t-SNE降维用于可视化
            tsne = TSNE(n_components=2, random_state=CONFIG['random_state'], perplexity=30, max_iter=1000)
            tsne_results = tsne.fit_transform(sbert_vectors)
            
            # 6. 创建图表
            plt.figure(figsize=(12, 10))
            
            # 绘制聚类结果
            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, 
                                cmap='viridis', s=60, alpha=0.8)
            
            # 添加聚类中心（在降维空间中手动计算）
            cluster_centers_tsne = np.zeros((best_n_clusters, 2))
            for cluster_id in range(best_n_clusters):
                # 找到属于当前聚类的点
                cluster_points = tsne_results[clusters == cluster_id]
                if len(cluster_points) > 0:
                    # 计算聚类中心
                    cluster_centers_tsne[cluster_id] = np.mean(cluster_points, axis=0)
            
            # 绘制聚类中心
            plt.scatter(cluster_centers_tsne[:, 0], cluster_centers_tsne[:, 1], 
                       c='red', marker='X', s=200, label='Cluster Centers')
            
            # 添加标题和坐标轴标签
            plt.title(f'SBERT Clustering Analysis (Agent: {agent_type}, Generation: {min_gen})\n'
                     f'Optimal Clusters: {best_n_clusters}, Average Intra-cluster Distance: {avg_intra_cluster_distance:.4f}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            # 添加图例
            plt.legend()
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('Cluster ID')
            
            # 添加网格
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            output_path = os.path.join(CONFIG['output_dir'], f'sbert_clustering_{agent_type}_generation_{min_gen}.png')
            plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
            plt.close()
        else:
            print(f"警告：{agent_type}的generation {min_gen}只有{len(gen_data)}个策略，跳过聚类分析")
        
        # 处理所有alive的策略数据
        alive_data = [d for d in agent_data if d['meta_data']['alive']]
        
        if len(alive_data) >= 2:
            # 绘制所有alive策略的聚类分析图
            # 提取SBERT向量
            sbert_vectors = np.array([d['meta_data']['sbert'] for d in alive_data])
            
            # 3. 聚类分析
            # 尝试不同的聚类数量，选择最优的
            best_n_clusters = 2
            best_silhouette = -1
            
            for n_clusters in range(2, min(10, len(alive_data)) + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=CONFIG['random_state'])
                clusters = kmeans.fit_predict(sbert_vectors)
                
                # 计算轮廓系数
                silhouette = silhouette_score(sbert_vectors, clusters)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_n_clusters = n_clusters
            
            # 使用最优聚类数量
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=CONFIG['random_state'])
            clusters = kmeans.fit_predict(sbert_vectors)
            
            # 4. 计算类内距离
            cluster_distances = []
            for cluster_id in range(best_n_clusters):
                cluster_vectors = sbert_vectors[clusters == cluster_id]
                if len(cluster_vectors) > 1:
                    # 计算类内平均距离
                    centroid = cluster_vectors.mean(axis=0)
                    distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
                    avg_distance = np.mean(distances)
                    cluster_distances.append(avg_distance)
            
            avg_intra_cluster_distance = np.mean(cluster_distances) if cluster_distances else 0
            
            # 5. t-SNE降维用于可视化
            tsne = TSNE(n_components=2, random_state=CONFIG['random_state'], perplexity=30, max_iter=1000)
            tsne_results = tsne.fit_transform(sbert_vectors)
            
            # 6. 创建图表
            plt.figure(figsize=(12, 10))
            
            # 绘制聚类结果
            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, 
                                cmap='viridis', s=60, alpha=0.8)
            
            # 添加聚类中心（在降维空间中手动计算）
            cluster_centers_tsne = np.zeros((best_n_clusters, 2))
            for cluster_id in range(best_n_clusters):
                # 找到属于当前聚类的点
                cluster_points = tsne_results[clusters == cluster_id]
                if len(cluster_points) > 0:
                    # 计算聚类中心
                    cluster_centers_tsne[cluster_id] = np.mean(cluster_points, axis=0)
            
            # 绘制聚类中心
            plt.scatter(cluster_centers_tsne[:, 0], cluster_centers_tsne[:, 1], 
                       c='red', marker='X', s=200, label='Cluster Centers')
            
            # 添加标题和坐标轴标签
            plt.title(f'SBERT Clustering Analysis (Agent: {agent_type}, Alive Strategies)\n'
                     f'Optimal Clusters: {best_n_clusters}, Average Intra-cluster Distance: {avg_intra_cluster_distance:.4f}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            # 添加图例
            plt.legend()
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('Cluster ID')
            
            # 添加网格
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            output_path = os.path.join(CONFIG['output_dir'], f'sbert_clustering_{agent_type}_alive.png')
            plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
            plt.close()
        else:
            print(f"警告：{agent_type}只有{len(alive_data)}个alive策略，跳过聚类分析")
    
    print("SBERT聚类分析图绘制完成")


def main() -> None:
    """主函数，依次调用各个模块"""
    print("开始ESB数据分析...")
    
    # 1. 数据预处理
    processed_data = preprocess_data()
    
    # 2. 信息熵计算
    df_entropy = calculate_entropy_by_generation(processed_data)
    
    # 3. t-SNE数据准备
    df_tsne = prepare_tsne_data(processed_data)
    
    # 4. 绘制图表
    plot_fitness_entropy_trend(processed_data)
    plot_tsne_visualization(df_tsne)
    plot_fitness_distribution(processed_data)
    plot_sbert_clustering(processed_data)
    
    print("ESB数据分析完成！")
    print(f"所有结果已保存到目录: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
