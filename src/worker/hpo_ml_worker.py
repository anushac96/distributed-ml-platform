import asyncio
import logging
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from .production_ml_worker import ProductionMLWorker

logger = logging.getLogger(__name__)

class HPOMLWorker(ProductionMLWorker):
    """ML Worker with hyperparameter optimization support"""
    
    def __init__(self, worker_id: str, config, hyperparameters: Dict[str, Any], 
                 dataset: str = "mnist", model_name: str = "cnn"):
        super().__init__(worker_id, config, dataset, model_name)
        self.hyperparameters = hyperparameters
        self._apply_hyperparameters()
    
    def _apply_hyperparameters(self):
        """Apply hyperparameters to model and training"""
        # Apply learning rate
        if 'learning_rate' in self.hyperparameters:
            self.learning_rate = self.hyperparameters['learning_rate']
        else:
            self.learning_rate = 0.001
        
        # Apply batch size
        if 'batch_size' in self.hyperparameters:
            self.batch_size = self.hyperparameters['batch_size']
            # Update data module batch size
            if hasattr(self, 'data_module') and self.data_module:
                self.data_module.batch_size = self.batch_size
        
        # Create optimizer with new learning rate
        self._create_optimizer()
    
    def _create_optimizer(self):
        """Create optimizer based on hyperparameters"""
        optimizer_type = self.hyperparameters.get('optimizer', 'adam')
        lr = self.hyperparameters.get('learning_rate', 0.001)
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            momentum = self.hyperparameters.get('momentum', 0.9)
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    async def train_with_hpo(self, trial_id: str) -> Dict[str, float]:
        """Train model and return metrics for HPO"""
        logger.info(f"Starting HPO training for trial {trial_id} with params: {self.hyperparameters}")
        
        try:
            # Initialize connections
            await self.connect_to_master()
            await self.connect_to_parameter_server()
            await self.register_with_master()
            await self.register_with_ps()
            
            # Initialize tracking variables
            best_accuracy = 0.0
            best_loss = float('inf')
            
            # Train for limited epochs (HPO typically uses fewer epochs)
            max_epochs = self.hyperparameters.get('max_epochs', 2)
            
            for epoch in range(max_epochs):
                # Training epoch
                train_loss, train_acc = await self._train_epoch_hpo()
                
                # Mock validation (simplified)
                val_loss, val_acc = train_loss * 0.8, train_acc * 0.95
                
                # Track best metrics
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                if val_loss < best_loss:
                    best_loss = val_loss
                
                logger.info(f"Trial {trial_id} Epoch {epoch+1}/{max_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Return metrics for optimization
            metrics = {
                'accuracy': best_accuracy,
                'loss': best_loss
            }
            
            logger.info(f"HPO training completed for trial {trial_id}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"HPO training failed for trial {trial_id}: {e}")
            return {'accuracy': 0.0, 'loss': 1.0}
            
        finally:
            # Cleanup without blocking
            logger.info(f"HPO training cleanup starting for trial {trial_id}")
            try:
                # Schedule cleanup asynchronously without awaiting
                asyncio.create_task(self._safe_cleanup())
                logger.info(f"HPO training cleanup scheduled for trial {trial_id}")
            except Exception as cleanup_error:
                logger.error(f"Cleanup scheduling failed: {cleanup_error}")
    
    async def _safe_cleanup(self):
        """Safe, non-blocking cleanup"""
        try:
            # Close connections gracefully
            if self.master_channel:
                await self.master_channel.close()
            if self.ps_channel:
                await self.ps_channel.close()
            logger.debug("HPO worker cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def _train_epoch_hpo(self) -> Tuple[float, float]:
        """Simplified training epoch for HPO"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        try:
            # Get training data (simplified)
            dataloader = self.data_module.get_train_dataloader(self.worker_id, 1)
            
            for batch_idx, (data, targets) in enumerate(dataloader):
                if batch_count >= 10:  # Limit batches for faster HPO
                    break
                    
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                batch_count += 1
            
            if batch_count > 0:
                return total_loss / batch_count, 100.0 * correct / total
            else:
                return 1.0, 0.0
                
        except Exception as e:
            logger.error(f"Training epoch failed: {e}")
            return 1.0, 0.0