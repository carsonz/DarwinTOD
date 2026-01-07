#!/usr/bin/env python3
"""
处理MultiWOZ数据集中的goal description字段，
删除"task "后面的编号和冒号，并删除每个字母间的". "
"""

import json
import re
import os
from tqdm import tqdm

def process_description(description):
    """
    处理description字符串：
    1. 删除"task "后面的编号和冒号
    2. 删除每个字母间的". "
    """
    # 第一步：删除"task "后面的编号和冒号
    # 匹配 "T. a. s. k.  .  数字. :.  " 模式
    description = re.sub(r'T\. a\. s\. k\.  \.  \d+\. :\.  ', '', description)
    
    # 第二步：删除每个字母间的". "
    # 匹配 "字母. " 模式，只保留字母
    description = re.sub(r'([a-zA-Z])\. ', r'\1', description)
    
    # 处理可能存在的特殊字符组合
    description = re.sub(r'\. ', '', description)  # 删除剩余的". "
    
    # 第三步：删除"Task XXXXX:"格式
    description = re.sub(r'Task \d+: ', '', description)
    
    return description

def process_json_file(input_file, output_file):
    """
    处理JSON文件，修改所有goal description字段
    """
    print(f"Loading JSON file from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} dialogues...")
    
    modified_count = 0
    
    # 使用tqdm显示进度
    for dialogue in tqdm(data, desc="Processing dialogues"):
        if 'goal' in dialogue and 'description' in dialogue['goal']:
            original_desc = dialogue['goal']['description']
            
            # 检查是否包含"T. a. s. k. "模式
            if 'T. a. s. k. ' in original_desc:
                # 处理description
                new_desc = process_description(original_desc)
                
                # 如果有变化，更新并计数
                if new_desc != original_desc:
                    dialogue['goal']['description'] = new_desc
                    modified_count += 1
    
    print(f"Modified {modified_count} descriptions")
    
    # 保存处理后的数据
    print(f"Saving processed data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Processing completed!")

def main():
    # 设置文件路径
    input_file = "./data/dialogues.json"
    output_file = "./data/dialogues_processed.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return
    
    # 处理文件
    process_json_file(input_file, output_file)
    
    print(f"Processed file saved as: {output_file}")

if __name__ == "__main__":
    main()