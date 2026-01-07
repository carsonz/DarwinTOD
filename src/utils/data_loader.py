"""
Data loader utility for MultiWOZ and SGD datasets.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple


class DataLoader:
    """Utility class for loading and processing dialogue data."""
    
    def __init__(self, dataset_type: str, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            dataset_type: Type of dataset ("multiwoz" or "sgd")
            data_dir: Base directory for data files
        """
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.dialogues = []
        
        # Set dataset-specific paths
        if dataset_type == "multiwoz":
            self.data_path = os.path.join(data_dir, "multiwoz21/data")
            self.dialogue_file = os.path.join(self.data_path, "dialogues.json")
        elif dataset_type == "sgd":
            self.data_path = os.path.join(data_dir, "sgd/data")
            self.dialogue_file = os.path.join(self.data_path, "dialogues.json")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def load_dialogues(self, data_split: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load dialogues from the dataset, optionally filtered by data_split.
        
        Args:
            data_split: Optional data split to filter by ("train", "validation", "test") or None for all dialogues
            
        Returns:
            List of dialogue dictionaries
        """
        with open(self.dialogue_file, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
        
        # Filter dialogues by data_split if specified
        if data_split is not None:
            self.dialogues = [d for d in dialogues if d.get("data_split") == data_split]
        else:
            self.dialogues = dialogues
        
        return self.dialogues
    
    def get_random_dialogues(self, n: int, data_split: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a random sample of dialogues, optionally filtered by data_split.
        
        Args:
            n: Number of dialogues to sample
            data_split: Optional data split to filter by ("train", "validation", "test") or None for all dialogues
            
        Returns:
            List of randomly selected dialogue dictionaries
        """
        import random
        self.dialogues = self.load_dialogues(data_split=data_split)
        return random.sample(self.dialogues, min(n, len(self.dialogues)))
    
    def extract_dialogue_examples(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract examples for strategy generation from dialogues.
        
        Args:
            dialogues: List of dialogue dictionaries
            
        Returns:
            List of example dictionaries with relevant information
        """
        examples = []
        
        for dialogue in dialogues:
            # Extract basic information
            example = {
                "dialogue_id": dialogue["dialogue_id"],
                "domains": dialogue["domains"],
                "turns": []
            }
            
            # Process each turn
            for turn in dialogue["turns"]:
                turn_data = {
                    "speaker": turn["speaker"],
                    "utterance": turn["utterance"]
                }
                
                # Add state information for user turns
                if turn["speaker"] == "user" and "state" in turn:
                    turn_data["state"] = turn["state"]
                
                # Add dialogue acts if available
                if "dialogue_acts" in turn:
                    turn_data["dialogue_acts"] = turn["dialogue_acts"]
                
                example["turns"].append(turn_data)
            
            examples.append(example)
        
        return examples
    
    def get_domains(self) -> List[str]:
        """
        Get all available domains for the dataset.
        
        Returns:
            List of domain names
        """
        if self.dataset_type == "multiwoz":
            return ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "police"]
        elif self.dataset_type == "sgd":
            # Get unique domains from dialogues
            domains = set()
            for dialogue in self.dialogues:
                domains.update(dialogue["domains"])
            return list(domains)
        else:
            return []
    
    def load_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Load a sample of dialogues for strategy generation.
        
        Args:
            num_samples: Number of dialogue samples to load
            
        Returns:
            List of dialogue dictionaries
        """
        # Get random dialogues
        self.get_random_dialogues(num_samples)
        
        # Extract examples for strategy generation
        return self.extract_dialogue_examples(self.dialogues)
    
    def get_initial_user_utterance(self, dialogue_id: str) -> Optional[str]:
        """
        Get the first user utterance from a specific dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            The first user utterance text, or None if not found
        """
        
        # Find the dialogue with the specified ID
        target_dialogue = None
        for dialogue in self.dialogues:
            if dialogue["dialogue_id"] == dialogue_id:
                target_dialogue = dialogue
                break
        
        if target_dialogue is None:
            return None
        
        # Find the first user turn
        for turn in target_dialogue["turns"]:
            if turn["speaker"] == "user":
                return turn["utterance"]
        
        return None