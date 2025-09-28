import logging
from typing import Dict, Any, List
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Track HPO experiments and results"""
    
    def __init__(self, storage_path: str = "./hpo_experiments"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_experiment(self, experiment_id: str, data: Dict[str, Any]):
        """Save experiment data"""
        file_path = os.path.join(self.storage_path, f"{experiment_id}.json")
        
        # Add timestamp
        data['saved_at'] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved experiment data: {file_path}")
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment data"""
        file_path = os.path.join(self.storage_path, f"{experiment_id}.json")
        
        if not os.path.exists(file_path):
            return {}
        
        with open(file_path, 'r') as f:
            return json.load(f)