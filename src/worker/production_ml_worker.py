import asyncio
import logging
import grpc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import time
from typing import Optional, Dict, Any

from ..generated import parameter_server_pb2
from ..generated import parameter_server_pb2_grpc
from ..generated import master_pb2
from ..generated import master_pb2_grpc
from ..utils.config import ConfigManager
from ..data.distributed_loader import MNISTDataModule, CIFAR10DataModule
from ..models.production_models import get_model
from ..checkpoint.checkpoint_manager import DistributedCheckpointManager

logger = logging.getLogger(__name__)

class ProductionMLWorker:
    """Production ML Worker with real datasets and models"""
    
    def __init__(self, worker_id: str, config: ConfigManager, 
                 dataset: str = "mnist", model_name: str = "cnn"):
        self.worker_id = worker_id
        self.config = config
        self.dataset = dataset
        self.model_name = model_name
        self.running = False
        
        # gRPC connections
        self.master_channel = None
        self.master_stub = None
        self.ps_channel = None
        self.ps_stub = None
        
        # ML components
        self.model = get_model(model_name, dataset)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Data loading
        if dataset == "mnist":
            self.data_module = MNISTDataModule(batch_size=32)
        elif dataset == "cifar10":
            self.data_module = CIFAR10DataModule(batch_size=32)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # Checkpointing
        self.checkpoint_manager = DistributedCheckpointManager(worker_id=worker_id)
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = 10
        self.current_version = {}
        self.training_metrics = {
            'epoch_losses': [],
            'batch_losses': [],
            'epoch_accuracies': []
        }
        
        logger.info(f"Production ML Worker {worker_id} initialized")
        logger.info(f"Dataset: {dataset}, Model: {model_name}")
    
    async def start(self):
        """Start production ML worker"""
        self.running = True
        logger.info(f"Starting Production ML Worker: {self.worker_id}")
        
        # Prepare data
        self.data_module.prepare_data()
        
        # Connect to services
        await self.connect_to_master()
        await self.connect_to_parameter_server()
        
        # Register with services
        await self.register_with_master()
        await self.register_with_ps()
        
        # Load checkpoint if exists
        await self.load_checkpoint()
        
        # Start training with real data
        await self.production_training_loop()
    
    async def connect_to_master(self):
        """Connect to master coordinator"""
        master_addr = f"{self.config.worker.master_host}:{self.config.worker.master_port}"
        self.master_channel = grpc.aio.insecure_channel(master_addr)
        self.master_stub = master_pb2_grpc.MasterServiceStub(self.master_channel)
        logger.info(f"Connected to master at {master_addr}")
    
    async def connect_to_parameter_server(self):
        """Connect to parameter server"""
        ps_addr = f"{self.config.parameter_server.host}:{self.config.parameter_server.port}"
        self.ps_channel = grpc.aio.insecure_channel(ps_addr)
        self.ps_stub = parameter_server_pb2_grpc.ParameterServerServiceStub(self.ps_channel)
        logger.info(f"Connected to parameter server at {ps_addr}")
    
    async def register_with_master(self):
        """Register with master coordinator"""
        capabilities = master_pb2.WorkerCapabilities(
            gpu_enabled=torch.cuda.is_available(),
            max_batch_size=64,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            supported_models=[self.model_name]
        )
        
        request = master_pb2.RegisterWorkerRequest(
            worker_id=self.worker_id,
            capabilities=capabilities,
            host="localhost",
            port=0
        )
        
        response = await self.master_stub.RegisterWorker(request)
        if response.success:
            logger.info(f"Registered with master: {response.message}")
    
    async def register_with_ps(self):
        """Register with parameter server"""
        request = parameter_server_pb2.ServerRegistrationRequest(
            worker_id=self.worker_id
        )
        
        response = await self.ps_stub.RegisterWorker(request)
        if response.success:
            logger.info(f"Registered with parameter server")
            
            # Initialize current versions for all model layers
            for name, param in self.model.named_parameters():
                self.current_version[name] = 0
    
    async def load_checkpoint(self):
        """Load checkpoint if available"""
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(latest_checkpoint)
            if checkpoint_data:
                self.model.load_state_dict(checkpoint_data['model_state'])
                self.optimizer.load_state_dict(checkpoint_data.get('optimizer_state', {}))
                self.current_epoch = checkpoint_data.get('epoch', 0)
                logger.info(f"Resumed from epoch {self.current_epoch}")
    
    async def pull_parameters(self, layer_name: str):
        """Pull latest parameters from parameter server"""
        request = parameter_server_pb2.PullRequest(
            worker_id=self.worker_id,
            layer_name=layer_name,
            current_version=self.current_version.get(layer_name, 0)
        )
        
        try:
            response = await self.ps_stub.PullParameters(request)
            if response.has_update and response.version > self.current_version.get(layer_name, 0):
                parameters = pickle.loads(response.parameter_data)
                self.update_layer_parameters(layer_name, parameters)
                self.current_version[layer_name] = response.version
                logger.debug(f"Updated {layer_name} to version {response.version}")
        except Exception as e:
            logger.error(f"Error pulling parameters for {layer_name}: {e}")
    
    async def push_gradients(self, layer_name: str, gradients: np.ndarray):
        """Push gradients to parameter server"""
        gradient_data = pickle.dumps(gradients)
        
        request = parameter_server_pb2.GradientUpdate(
            worker_id=self.worker_id,
            layer_name=layer_name,
            gradient_data=gradient_data,
            version=self.current_version.get(layer_name, 0)
        )
        
        try:
            response = await self.ps_stub.PushGradients(request)
            if response.success:
                logger.debug(f"Pushed gradients for {layer_name}")
        except Exception as e:
            logger.error(f"Error pushing gradients for {layer_name}: {e}")
    
    def update_layer_parameters(self, layer_name: str, parameters: np.ndarray):
        """Update model layer with new parameters"""
        try:
            # Find the parameter in the model
            for name, param in self.model.named_parameters():
                if name == layer_name:
                    if param.data.numel() == parameters.size:
                        param.data = torch.from_numpy(parameters.reshape(param.data.shape))
                        break
        except Exception as e:
            logger.error(f"Error updating layer {layer_name}: {e}")
    
    def extract_layer_gradients(self, layer_name: str) -> Optional[np.ndarray]:
        """Extract gradients from specific layer"""
        for name, param in self.model.named_parameters():
            if name == layer_name and param.grad is not None:
                return param.grad.data.cpu().numpy().flatten()
        return None
    
    def calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate prediction accuracy"""
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return 100 * correct / total
    
    async def production_training_loop(self):
        """Main production training loop with real data"""
        logger.info(f"Starting production training: {self.dataset} dataset, {self.model_name} model")
        
        # Get distributed dataloader
        num_workers = 2  # Assuming 2 workers for now
        train_dataloader = self.data_module.get_train_dataloader(self.worker_id, num_workers)
        
        for epoch in range(self.current_epoch, self.total_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.total_epochs}")
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            batch_count = 0
            
            # Set epoch for distributed sampler
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            
            for batch_idx, (data, target) in enumerate(train_dataloader):
                if not self.running:
                    break
                
                # Pull latest parameters
                for layer_name in self.current_version.keys():
                    await self.pull_parameters(layer_name)
                
                # Forward pass
                self.optimizer.zero_grad()

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate accuracy
                accuracy = self.calculate_accuracy(output, target)
                
                # Backward pass
                loss.backward()
                
                # Push gradients to parameter server
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients = self.extract_layer_gradients(name)
                        if gradients is not None:
                            await self.push_gradients(name, gradients)
                
                # Update local optimizer (for momentum, etc.)
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                batch_count += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Batch {batch_idx}: Loss={loss.item():.4f}, Acc={accuracy:.2f}%")
                
                await asyncio.sleep(0.1)  # Allow other async operations
            
            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            avg_epoch_accuracy = epoch_accuracy / batch_count if batch_count > 0 else 0
            
            self.training_metrics['epoch_losses'].append(avg_epoch_loss)
            self.training_metrics['epoch_accuracies'].append(avg_epoch_accuracy)
            
            logger.info(f"Epoch {epoch + 1} completed: Loss={avg_epoch_loss:.4f}, Acc={avg_epoch_accuracy:.2f}%")
            
            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                await self.save_checkpoint(epoch + 1, avg_epoch_loss)
            
            self.current_epoch = epoch + 1
        
        logger.info("Production training completed")
        await self.final_evaluation()
    
    async def save_checkpoint(self, epoch: int, loss: float):
        """Save training checkpoint"""
        try:
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()
            
            metadata = {
                'dataset': self.dataset,
                'model_name': self.model_name,
                'training_metrics': self.training_metrics
            }
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model_state=model_state,
                optimizer_state=optimizer_state,
                epoch=epoch,
                loss=loss,
                metadata=metadata
            )
            
            if checkpoint_path:
                logger.info(f"Checkpoint saved at epoch {epoch}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    async def final_evaluation(self):
        """Run final evaluation on test set"""
        logger.info("Running final evaluation...")
        
        test_dataloader = self.data_module.get_test_dataloader()
        self.model.eval()
        
        test_loss = 0
        test_accuracy = 0
        batch_count = 0
        
        with torch.no_grad():
            for data, target in test_dataloader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                test_accuracy += self.calculate_accuracy(output, target)
                batch_count += 1
                
                if batch_count >= 10:  # Limit evaluation batches
                    break
        
        avg_test_loss = test_loss / batch_count
        avg_test_accuracy = test_accuracy / batch_count
        
        logger.info(f"Final Evaluation - Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.2f}%")
    
    async def stop(self):
        """Stop worker and save final checkpoint"""
        self.running = False
        await self.save_checkpoint(self.current_epoch, 0.0)
        
        if self.master_channel:
            await self.master_channel.close()
        if self.ps_channel:
            await self.ps_channel.close()