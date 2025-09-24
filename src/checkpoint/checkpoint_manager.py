import os
import torch
import pickle
import json
import logging
from typing import Dict, Optional, Any, List
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages model checkpoints and training state"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoint manager initialized: {checkpoint_dir}")
    
    def save_checkpoint(self, 
                       model_state: Dict[str, Any],
                       optimizer_state: Optional[Dict] = None,
                       epoch: int = 0,
                       loss: float = 0.0,
                       metadata: Optional[Dict] = None) -> str:
        """Save model checkpoint"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint_data = {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'epoch': epoch,
            'loss': loss,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            
            # Save metadata separately for easy inspection
            metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'loss': loss,
                    'timestamp': timestamp,
                    'checkpoint_path': checkpoint_path,
                    'metadata': metadata or {}
                }, f, indent=2)
            
            logger.info(f"Checkpoint saved: {checkpoint_name}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return ""
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load model checkpoint"""
        try:
            if os.path.exists(checkpoint_path):
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                logger.info(f"Checkpoint loaded: {checkpoint_path}")
                return checkpoint_data
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint"""
        try:
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                              if f.endswith('.pt') and f.startswith('checkpoint_')]
            
            if not checkpoint_files:
                return None
            
            # Sort by modification time
            checkpoint_files.sort(key=lambda f: os.path.getmtime(
                os.path.join(self.checkpoint_dir, f)
            ), reverse=True)
            
            latest_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_files[0])
            logger.info(f"Latest checkpoint: {latest_checkpoint}")
            return latest_checkpoint
            
        except Exception as e:
            logger.error(f"Error finding latest checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata"""
        checkpoints = []
        
        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('_metadata.json'):
                    metadata_path = os.path.join(self.checkpoint_dir, filename)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        checkpoints.append(metadata)
            
            # Sort by epoch
            checkpoints.sort(key=lambda x: x.get('epoch', 0), reverse=True)
            return checkpoints
            
        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}")
            return []
    
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """Remove old checkpoints, keeping only the most recent ones"""
        try:
            checkpoints = self.list_checkpoints()
            
            if len(checkpoints) <= keep_count:
                return
            
            # Remove old checkpoints
            for checkpoint_info in checkpoints[keep_count:]:
                checkpoint_path = checkpoint_info.get('checkpoint_path', '')
                metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
                
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    
                logger.info(f"Removed old checkpoint: {os.path.basename(checkpoint_path)}")
                
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")

class DistributedCheckpointManager(CheckpointManager):
    """Checkpoint manager for distributed training"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", worker_id: str = "worker_0"):
        super().__init__(checkpoint_dir)
        self.worker_id = worker_id
        self.worker_dir = os.path.join(checkpoint_dir, worker_id)
        os.makedirs(self.worker_dir, exist_ok=True)
    
    def save_worker_checkpoint(self, model_state: Dict, epoch: int, loss: float) -> str:
        """Save checkpoint specific to this worker"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{self.worker_id}_checkpoint_epoch_{epoch}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.worker_dir, checkpoint_name)
        
        checkpoint_data = {
            'worker_id': self.worker_id,
            'model_state': model_state,
            'epoch': epoch,
            'loss': loss,
            'timestamp': timestamp
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Worker checkpoint saved: {checkpoint_name}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Error saving worker checkpoint: {e}")
            return ""