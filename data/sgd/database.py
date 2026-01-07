import json
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
        data_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
        self.domains = ['Alarm_1', 'Banks_1', 'Banks_2', 'Buses_1', 'Buses_2', 'Buses_3', 'Calendar_1', 'Events_1', 'Events_2', 'Events_3', 'Flights_1', 'Flights_2', 'Flights_3', 'Flights_4', 'Homes_1', 'Homes_2', 'Hotels_1', 'Hotels_2', 'Hotels_3', 'Hotels_4', 'Media_1', 'Media_2', 'Media_3', 'Messaging_1', 'Movies_1', 'Movies_2', 'Movies_3', 'Music_1', 'Music_2', 'Music_3', 'Payment_1', 'RentalCars_1', 'RentalCars_2', 'RentalCars_3', 'Restaurants_1', 'Restaurants_2', 'RideSharing_1', 'RideSharing_2', 'Services_1', 'Services_2', 'Services_3', 'Services_4', 'Trains_1', 'Travel_1', 'Weather_1']
        self.dbs = {}
        
        # 加载各domain的数据库文件
        for domain in self.domains:
            db_file = os.path.join(data_path, f"{domain.lower()}_db.json")
            if os.path.exists(db_file):
                with open(db_file, 'r') as f:
                    self.dbs[domain] = json.load(f)
            else:
                print(f"Warning: Database file for domain {domain} not found at {db_file}")
                self.dbs[domain] = []

    def query(self, domain: str, state: dict, topk: int, ignore_open=False, soft_contraints=(), fuzzy_match_ratio=60) -> list:
        """
        返回基于对话状态的topk个实体列表
        :param domain: 查询的domain
        :param state: 支持两种格式: 1) [[slot,value], [slot,value]...]; 2) {domain: {slot: value, slot: value...}}
        :param topk: 返回的结果数量
        :param ignore_open: 是否忽略开放约束
        :param soft_contraints: 软约束
        :param fuzzy_match_ratio: 模糊匹配阈值
        :return: 匹配的实体列表
        """
        if isinstance(state, dict):
            assert domain in state, print(f"domain {domain} not in state {state}")
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
                res['Ref'] = '{0:08d}'.format(i)
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
            print(f"Test query for {test_domain} returned {len(res)} results")
            if res:
                print("First result:", res[0])
        else:
            print(f"No data available for domain {test_domain}")
