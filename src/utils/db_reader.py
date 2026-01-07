"""
Database reader utility for MultiWOZ and SGD datasets.
"""

import json
import os
from typing import Dict, List, Any, Optional


class DatabaseReader:
    """Utility class for reading and querying database files from MultiWOZ and SGD datasets."""
    
    def __init__(self, dataset_type: str, data_dir: str = "data"):
        """
        Initialize the database reader.
        
        Args:
            dataset_type: Type of dataset ("multiwoz" or "sgd")
            data_dir: Base directory for data files
        """
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        
        # Set dataset-specific paths
        if dataset_type == "multiwoz":
            self.db_dir = os.path.join(data_dir, "multiwoz21/data")
            self.dialogue_file = os.path.join(self.db_dir, "dialogues.json")
        elif dataset_type == "sgd":
            self.db_dir = os.path.join(data_dir, "sgd/data")
            self.dialogue_file = os.path.join(self.db_dir, "dialogues.json")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def load_dialogues(self) -> List[Dict[str, Any]]:
        """
        Load all dialogues from the dataset.
        
        Returns:
            List of dialogue dictionaries
        """
        with open(self.dialogue_file, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
        return dialogues
    
    def get_random_dialogues(self, n: int) -> List[Dict[str, Any]]:
        """
        Get a random sample of dialogues.
        
        Args:
            n: Number of dialogues to sample
            
        Returns:
            List of randomly selected dialogue dictionaries
        """
        import random
        all_dialogues = self.load_dialogues()
        return random.sample(all_dialogues, min(n, len(all_dialogues)))
    
    def get_database(self, domain: str) -> List[Dict[str, Any]]:
        """
        Load database entries for a specific domain.
        
        Args:
            domain: Domain name (e.g., "hotel", "restaurant" for MultiWOZ,
                   "Hotels_1", "Restaurants_1" for SGD)
                   
        Returns:
            List of database entries
        """
        # Determine database file name based on dataset type
        if self.dataset_type == "multiwoz":
            db_file = os.path.join(self.db_dir, f"{domain}_db.json")
        elif self.dataset_type == "sgd":
            db_file = os.path.join(self.db_dir, f"{domain.lower()}_db.json")
        
        if not os.path.exists(db_file):
            raise FileNotFoundError(f"Database file not found: {db_file}")
        
        with open(db_file, 'r', encoding='utf-8') as f:
            db_data = json.load(f)
        return db_data
    
    def query_database(self, domain: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the database with constraints.
        
        Args:
            domain: Domain name
            constraints: Dictionary of field-value constraints
            
        Returns:
            List of matching database entries
        """
        db_entries = self.get_database(domain)
        results = []
        
        for entry in db_entries:
            match = True
            for field, value in constraints.items():
                if field not in entry or entry[field] != value:
                    match = False
                    break
            
            if match:
                results.append(entry)
        
        return results
    
    def get_domains(self) -> List[str]:
        """
        Get all available domains for the dataset.
        
        Returns:
            List of domain names
        """
        if self.dataset_type == "multiwoz":
            return ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "police"]
        elif self.dataset_type == "sgd":
            # Get all domain database files
            domains = []
            for file in os.listdir(self.db_dir):
                if file.endswith("_db.json"):
                    domain = file.replace("_db.json", "")
                    domains.append(domain)
            return domains
        else:
            return []