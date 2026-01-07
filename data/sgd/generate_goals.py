#!/usr/bin/env python3
"""
为SGD数据集生成goal信息的脚本
基于对话中的状态信息、意图和槽位值来构建goal描述
"""

import json
import os
import sys
from typing import Dict, List, Any

def load_schema(schema_path: str) -> Dict[str, Any]:
    """加载schema文件"""
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dialogues(dialogues_path: str) -> List[Dict[str, Any]]:
    """加载对话数据"""
    with open(dialogues_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_domain_schema(domain: str, schema_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """获取特定领域的schema信息"""
    for schema in schema_list:
        if schema["service_name"] == domain:
            return schema
    return {}

def get_intent_description(intent_name: str, domain_schema: Dict[str, Any]) -> str:
    """获取意图的描述"""
    for intent in domain_schema.get("intents", []):
        if intent["name"] == intent_name:
            return intent.get("description", "")
    return ""

def get_slot_description(slot_name: str, domain_schema: Dict[str, Any]) -> str:
    """获取槽位的描述"""
    for slot in domain_schema.get("slots", []):
        if slot["name"] == slot_name:
            return slot.get("description", "")
    return ""

def generate_goal_description(dialogue: Dict[str, Any], schema_list: List[Dict[str, Any]]) -> str:
    """生成goal描述"""
    domains = dialogue.get("domains", [])
    if not domains:
        return ""
    
    # 获取对话中的主要意图和槽位信息
    user_intents = {}
    slot_values = {}
    
    # 从对话轮次中提取意图和槽位值
    for turn in dialogue.get("turns", []):
        if turn.get("speaker") != "user":
            continue
            
        # 获取active_intent
        for domain, intent in turn.get("active_intent", {}).items():
            if intent:  # 只记录非空意图
                if domain not in user_intents:
                    user_intents[domain] = set()
                user_intents[domain].add(intent)
        
        # 获取state中的槽位值
        for domain, slots in turn.get("state", {}).items():
            if domain not in slot_values:
                slot_values[domain] = {}
            for slot, value in slots.items():
                if value and value.strip():  # 只记录非空值
                    slot_values[domain][slot] = value
    
    # 构建goal描述
    description_parts = []
    
    for domain in domains:
        domain_schema = get_domain_schema(domain, schema_list)
        domain_desc = domain_schema.get("description", "")
        
        if domain in user_intents and user_intents[domain]:
            # 获取该领域的主要意图
            intents = list(user_intents[domain])
            if len(intents) == 1:
                intent_desc = get_intent_description(intents[0], domain_schema)
                if intent_desc:
                    description_parts.append(f"You want to {intent_desc.lower()}")
                else:
                    description_parts.append(f"You want to {intents[0].lower()}")
            else:
                # 如果有多个意图，使用第一个作为主要意图
                intent_desc = get_intent_description(intents[0], domain_schema)
                if intent_desc:
                    description_parts.append(f"You want to {intent_desc.lower()}")
                else:
                    description_parts.append(f"You want to {intents[0].lower()}")
        
        # 添加具体的槽位值信息
        if domain in slot_values and slot_values[domain]:
            slot_desc_parts = []
            for slot, value in slot_values[domain].items():
                # 跳过一些常见的辅助槽位
                if slot in ["count"]:
                    continue
                    
                slot_desc = get_slot_description(slot, domain_schema)
                if slot_desc:
                    slot_desc_parts.append(f"{slot_desc} is <span class='emphasis'>{value}</span>")
                else:
                    slot_desc_parts.append(f"{slot} is <span class='emphasis'>{value}</span>")
            
            if slot_desc_parts:
                description_parts.append("Make sure you get " + ", ".join(slot_desc_parts))
    
    return ". ".join(description_parts) + "." if description_parts else ""

def generate_goal_inform_request(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """生成goal的inform和request部分"""
    domains = dialogue.get("domains", [])
    inform = {}
    request = {}
    
    # 从对话轮次中提取槽位值
    for turn in dialogue.get("turns", []):
        if turn.get("speaker") != "user":
            continue
            
        # 获取state中的槽位值
        for domain, slots in turn.get("state", {}).items():
            if domain not in inform:
                inform[domain] = {}
            for slot, value in slots.items():
                if value and value.strip():  # 只记录非空值
                    inform[domain][slot] = value
    
    # 从对话轮次中提取请求的槽位
    for turn in dialogue.get("turns", []):
        if turn.get("speaker") != "user":
            continue
            
        # 获取requested_slots
        for domain, slots in turn.get("requested_slots", {}).items():
            if domain not in request:
                request[domain] = {}
            for slot in slots:
                if slot and slot.strip():  # 只记录非空槽位
                    request[domain][slot] = ""
    
    return {"inform": inform, "request": request}

def process_dialogues(dialogues: List[Dict[str, Any]], schema_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """处理所有对话，添加goal信息"""
    processed_dialogues = []
    
    for dialogue in dialogues:
        # 复制原始对话
        new_dialogue = dialogue.copy()
        
        # 生成goal描述
        goal_description = generate_goal_description(dialogue, schema_list)
        goal_inform_request = generate_goal_inform_request(dialogue)
        
        # 更新goal
        new_dialogue["goal"] = {
            "description": goal_description,
            "inform": goal_inform_request["inform"],
            "request": goal_inform_request["request"]
        }
        
        processed_dialogues.append(new_dialogue)
    
    return processed_dialogues

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    schema_dir = os.path.join(current_dir, "dstc8-schema-guided-dialogue-master", "train")
    
    # 加载数据
    print("Loading schema...")
    schema_path = os.path.join(schema_dir, "schema.json")
    schema_list = load_schema(schema_path)
    
    print("Loading dialogues...")
    dialogues_path = os.path.join(data_dir, "dialogues.json")
    dialogues = load_dialogues(dialogues_path)
    
    print(f"Processing {len(dialogues)} dialogues...")
    processed_dialogues = process_dialogues(dialogues, schema_list)
    
    # 保存结果
    output_path = os.path.join(data_dir, "dialogues_with_goals.json")
    print(f"Saving processed dialogues to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_dialogues, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    main()