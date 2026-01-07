#!/usr/bin/env python3
"""
SGD数据集数据库提取脚本
从SGD数据集的dialogues.json中提取各个domain的数据库信息，
为每个domain创建独立的db.json文件，类似于MultiWOZ数据集的结构。
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set

# 添加convlab路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def extract_domains_from_dialogues(dialogues_file: str) -> Set[str]:
    """从对话数据中提取所有domains"""
    domains = set()
    with open(dialogues_file, 'r') as f:
        dialogues = json.load(f)
    
    for dialogue in dialogues:
        domains.update(dialogue['domains'])
    
    return domains

def extract_db_results_for_domain(dialogues_file: str, domain: str) -> List[Dict]:
    """从对话数据中提取特定domain的db_results"""
    db_results = []
    seen_entries = set()  # 用于去重
    
    with open(dialogues_file, 'r') as f:
        dialogues = json.load(f)
    
    for dialogue in dialogues:
        # 只处理包含目标domain的对话
        if domain not in dialogue['domains']:
            continue
            
        for turn in dialogue['turns']:
            if 'db_results' in turn and turn['db_results'] and domain in turn['db_results']:
                for entry in turn['db_results'][domain]:
                    # 创建一个用于去重的键，基于所有字段值
                    entry_key = tuple(sorted(entry.items()))
                    if entry_key not in seen_entries:
                        seen_entries.add(entry_key)
                        db_results.append(entry)
    
    return db_results

def save_domain_db(domain: str, db_data: List[Dict], output_dir: str):
    """保存domain的数据库到文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据库文件
    output_file = os.path.join(output_dir, f"{domain.lower()}_db.json")
    with open(output_file, 'w') as f:
        json.dump(db_data, f, indent=2)
    
    print(f"Saved {len(db_data)} entries for domain {domain} to {output_file}")

def create_database_py(domains: List[str], output_dir: str):
    """创建类似于MultiWOZ的database.py文件"""
    template = f'''import json
import os
import random
from fuzzywuzzy import fuzz
from itertools import chain
from zipfile import ZipFile
from copy import deepcopy
import sys

# 添加convlab路径
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from convlab.util.unified_datasets_util import BaseDatabase, download_unified_datasets


class Database(BaseDatabase):
    def __init__(self):
        """加载SGD数据集的数据库文件"""
        data_path = os.path.dirname(os.path.abspath(__file__))
        self.domains = {', '.join([f"'{d}'" for d in domains])}
        self.dbs = {{}}
        
        # 加载各domain的数据库文件
        for domain in self.domains:
            db_file = os.path.join(data_path, f"{{domain.lower()}}_db.json")
            if os.path.exists(db_file):
                with open(db_file, 'r') as f:
                    self.dbs[domain] = json.load(f)
            else:
                print(f"Warning: Database file for domain {{domain}} not found at {{db_file}}")
                self.dbs[domain] = []

    def query(self, domain: str, state: dict, topk: int, ignore_open=False, soft_contraints=(), fuzzy_match_ratio=60) -> list:
        """
        返回基于对话状态的topk个实体列表
        :param domain: 查询的domain
        :param state: 支持两种格式: 1) [[slot,value], [slot,value]...]; 2) {{domain: {{slot: value, slot: value...}}}}
        :param topk: 返回的结果数量
        :param ignore_open: 是否忽略开放约束
        :param soft_contraints: 软约束
        :param fuzzy_match_ratio: 模糊匹配阈值
        :return: 匹配的实体列表
        """
        if isinstance(state, dict):
            assert domain in state, print(f"domain {{domain}} not in state {{state}}")
            state = state[domain].items()
        
        # 如果domain不存在，返回空列表
        if domain not in self.dbs:
            return []
            
        found = []
        for i, record in enumerate(self.dbs[domain]):
            constraints_iterator = zip(state, [False] * len(state))
            soft_contraints_iterator = zip(soft_contraints, [True] * len(soft_contraints))
            for (key, val), fuzzy_match in chain(constraints_iterator, soft_contraints_iterator):
                if val in ["", "dont care", 'not mentioned', "don't care", "dontcare", "do n't care", "do not care"]:
                    pass
                else:
                    try:
                        if key not in record:
                            continue
                        if record[key].strip() == '?':
                            # '?' matches any constraint
                            continue
                        else:
                            if not fuzzy_match:
                                if val.strip().lower() != record[key].strip().lower():
                                    break
                            else:
                                if fuzz.partial_ratio(val.strip().lower(), record[key].strip().lower()) < fuzzy_match_ratio:
                                    break
                    except:
                        continue
            else:
                res = deepcopy(record)
                res['Ref'] = '{{0:08d}}'.format(i)
                found.append(res)
                if len(found) == topk:
                    return found
        return found


if __name__ == '__main__':
    db = Database()
    assert issubclass(Database, BaseDatabase)
    assert isinstance(db, BaseDatabase)
    # 测试查询
    if db.domains:
        test_domain = db.domains[0]
        if test_domain in db.dbs and db.dbs[test_domain]:
            res = db.query(test_domain, [], topk=3)
            print(f"Test query for {{test_domain}} returned {{len(res)}} results")
            if res:
                print("First result:", res[0])
        else:
            print(f"No data available for domain {{test_domain}}")
'''
    
    output_file = os.path.join(output_dir, "database.py")
    with open(output_file, 'w') as f:
        f.write(template)
    
    print(f"Created database.py at {output_file}")

def main():
    # 设置路径
    data_dir = os.path.dirname(os.path.abspath(__file__))
    dialogues_file = os.path.join(data_dir, "data", "dialogues.json")
    output_dir = data_dir
    
    # 提取所有domains
    print("Extracting domains from dialogues...")
    domains = extract_domains_from_dialogues(dialogues_file)
    print(f"Found {len(domains)} domains: {sorted(domains)}")
    
    # 为每个domain提取数据库信息
    for domain in sorted(domains):
        print(f"Extracting database for domain: {domain}")
        db_data = extract_db_results_for_domain(dialogues_file, domain)
        if db_data:
            save_domain_db(domain, db_data, output_dir)
        else:
            print(f"No database entries found for domain: {domain}")
    
    # 创建database.py文件
    print("Creating database.py file...")
    create_database_py(sorted(domains), output_dir)
    
    print("Database extraction completed!")

if __name__ == "__main__":
    main()