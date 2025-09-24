import logging
import time
from typing import Dict, List, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor distributed training performance"""
    
    def __init__(self):
        self.metrics = {
            'worker_metrics': {},
            'system_metrics': {
                'start_time': time.time(),
                'total_batches_processed': 0,
                'total_parameters_updated': 0
            },
            'performance_history': []
        }
    
    def record_worker_batch(self, worker_id: str, batch_time: float, 
                           loss: float, accuracy: float):
        """Record batch completion metrics"""
        if worker_id not in self.metrics['worker_metrics']:
            self.metrics['worker_metrics'][worker_id] = {
                'batches_completed': 0,
                'total_time': 0,
                'avg_loss': 0,
                'avg_accuracy': 0
            }
        
        worker_metrics = self.metrics['worker_metrics'][worker_id]
        worker_metrics['batches_completed'] += 1
        worker_metrics['total_time'] += batch_time
        
        # Running average
        n = worker_metrics['batches_completed']
        worker_metrics['avg_loss'] = ((n-1) * worker_metrics['avg_loss'] + loss) / n
        worker_metrics['avg_accuracy'] = ((n-1) * worker_metrics['avg_accuracy'] + accuracy) / n
        
        self.metrics['system_metrics']['total_batches_processed'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        current_time = time.time()
        elapsed_time = current_time - self.metrics['system_metrics']['start_time']
        
        summary = {
            'elapsed_time_minutes': elapsed_time / 60,
            'total_batches': self.metrics['system_metrics']['total_batches_processed'],
            'batches_per_minute': self.metrics['system_metrics']['total_batches_processed'] / (elapsed_time / 60) if elapsed_time > 0 else 0,
            'worker_performance': {}
        }
        
        for worker_id, metrics in self.metrics['worker_metrics'].items():
            if metrics['batches_completed'] > 0:
                summary['worker_performance'][worker_id] = {
                    'batches_completed': metrics['batches_completed'],
                    'avg_batch_time': metrics['total_time'] / metrics['batches_completed'],
                    'avg_loss': metrics['avg_loss'],
                    'avg_accuracy': metrics['avg_accuracy']
                }
        
        return summary
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': self.metrics,
                    'summary': self.get_performance_summary()
                }, f, indent=2)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")