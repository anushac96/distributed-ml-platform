import asyncio
import logging
import grpc
import torch
import torch.nn as nn
import numpy as np
import pickle
import time

# Use YOUR proto messages
from ..generated import parameter_server_pb2
from ..generated import parameter_server_pb2_grpc
from ..generated import master_pb2
from ..generated import master_pb2_grpc
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """Simple neural network for testing distributed training"""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class MLWorker:
    """ML Worker that trains with parameter server"""
    
    def __init__(self, worker_id: str, config: ConfigManager):
        self.worker_id = worker_id
        self.config = config
        self.running = False
        
        # gRPC connections
        self.master_channel = None
        self.master_stub = None
        self.ps_channel = None
        self.ps_stub = None
        
        # ML components
        self.model = SimpleModel()
        self.criterion = nn.CrossEntropyLoss()
        self.current_version = {"linear1": 0, "linear2": 0}
        
        logger.info(f"ML Worker {worker_id} initialized with simple neural network")
    
    async def start(self):
        """Start ML worker"""
        self.running = True
        logger.info(f"Starting ML Worker: {self.worker_id}")
        
        # Connect to services
        await self.connect_to_master()
        await self.connect_to_parameter_server()
        
        # Register with both services
        await self.register_with_master()
        await self.register_with_ps()
        
        # Start training
        await self.training_loop()
    
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
            gpu_enabled=False,
            max_batch_size=32,
            device_type="cpu",
            supported_models=["simple_neural_network"]
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
        """Register with parameter server using YOUR message format"""
        request = parameter_server_pb2.ServerRegistrationRequest(
            worker_id=self.worker_id
        )
        
        response = await self.ps_stub.RegisterWorker(request)
        if response.success:
            logger.info(f"Registered with parameter server. Available layers: {list(response.available_layers)}")
    
    async def pull_parameters(self, layer_name: str):
        """Pull latest parameters from server using YOUR message format"""
        request = parameter_server_pb2.PullRequest(
            worker_id=self.worker_id,
            layer_name=layer_name,
            current_version=self.current_version[layer_name]
        )
        
        response = await self.ps_stub.PullParameters(request)
        if response.has_update and response.version > self.current_version[layer_name]:
            # Update local model with new parameters
            parameters = pickle.loads(response.parameter_data)
            self.update_layer_parameters(layer_name, parameters)
            self.current_version[layer_name] = response.version
            logger.info(f"Updated {layer_name} to version {response.version}")
    
    async def push_gradients(self, layer_name: str, gradients: np.ndarray):
        """Push gradients to server using YOUR GradientUpdate message"""
        gradient_data = pickle.dumps(gradients)
        
        request = parameter_server_pb2.GradientUpdate(
            worker_id=self.worker_id,
            layer_name=layer_name,
            gradient_data=gradient_data,
            version=self.current_version[layer_name]
        )
        
        response = await self.ps_stub.PushGradients(request)
        if response.success:
            logger.debug(f"Pushed gradients for {layer_name}")
    
    def update_layer_parameters(self, layer_name: str, parameters: np.ndarray):
        """Update model layer with new parameters from parameter server"""
        try:
            if layer_name == "linear1":
                # Linear1: 128 neurons, 784 inputs
                weight_shape = (128, 784)
                bias_shape = (128,)
                
                if parameters.size >= np.prod(weight_shape) + np.prod(bias_shape):
                    weight_params = parameters[:np.prod(weight_shape)].reshape(weight_shape)
                    bias_params = parameters[np.prod(weight_shape):np.prod(weight_shape)+np.prod(bias_shape)]
                    
                    self.model.linear1.weight.data = torch.from_numpy(weight_params)
                    self.model.linear1.bias.data = torch.from_numpy(bias_params)
                    
            elif layer_name == "linear2":
                # Linear2: 10 neurons, 128 inputs  
                weight_shape = (10, 128)
                bias_shape = (10,)
                
                if parameters.size >= np.prod(weight_shape) + np.prod(bias_shape):
                    weight_params = parameters[:np.prod(weight_shape)].reshape(weight_shape)
                    bias_params = parameters[np.prod(weight_shape):np.prod(weight_shape)+np.prod(bias_shape)]
                    
                    self.model.linear2.weight.data = torch.from_numpy(weight_params)
                    self.model.linear2.bias.data = torch.from_numpy(bias_params)
                    
        except Exception as e:
            logger.error(f"Error updating {layer_name}: {e}")
    
    def extract_layer_gradients(self, layer_name: str) -> np.ndarray:
        """Extract gradients from a layer"""
        if layer_name == "linear1" and self.model.linear1.weight.grad is not None:
            weight_grad = self.model.linear1.weight.grad.numpy().flatten()
            bias_grad = self.model.linear1.bias.grad.numpy().flatten()
            return np.concatenate([weight_grad, bias_grad])
            
        elif layer_name == "linear2" and self.model.linear2.weight.grad is not None:
            weight_grad = self.model.linear2.weight.grad.numpy().flatten()
            bias_grad = self.model.linear2.bias.grad.numpy().flatten()
            return np.concatenate([weight_grad, bias_grad])
            
        return np.array([])  # Return empty if no gradients
    
    async def training_loop(self):
        """Main training loop - this is where distributed ML magic happens"""
        logger.info("Starting distributed training loop")
        
        for iteration in range(10):  # 10 training iterations
            logger.info(f"Training iteration {iteration + 1}/10")
            
            # Generate dummy training data (simulating MNIST)
            batch_data = torch.randn(32, 1, 28, 28)  
            batch_labels = torch.randint(0, 10, (32,))   
            
            # Pull latest parameters from parameter server
            await self.pull_parameters("linear1")
            await self.pull_parameters("linear2")
            
            # Forward pass
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            
            # Extract and push gradients to parameter server
            linear1_grad = self.extract_layer_gradients("linear1")
            linear2_grad = self.extract_layer_gradients("linear2")
            
            if linear1_grad.size > 0:
                await self.push_gradients("linear1", linear1_grad)
            if linear2_grad.size > 0:
                await self.push_gradients("linear2", linear2_grad)
            
            # Clear gradients
            self.model.zero_grad()
            
            logger.info(f"Completed iteration {iteration + 1}, Loss: {loss.item():.4f}")
            await asyncio.sleep(3)  # Wait between iterations
        
        logger.info("Distributed training completed")
    
    async def stop(self):
        """Stop worker"""
        self.running = False
        if self.master_channel:
            await self.master_channel.close()
        if self.ps_channel:
            await self.ps_channel.close()