import asyncio
import logging
import grpc
import numpy as np
import pickle
import time
from typing import Dict
from concurrent import futures

# Use YOUR proto messages (different names)
from ..generated import parameter_server_pb2
from ..generated import parameter_server_pb2_grpc
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

class ParameterStorage:
    """Stores model parameters and gradients"""
    
    def __init__(self):
        self.parameters: Dict[str, np.ndarray] = {}  # layer_name -> parameters
        self.gradients: Dict[str, list] = {}         # layer_name -> list of gradients
        self.versions: Dict[str, int] = {}           # layer_name -> version number
    
    def initialize_layer(self, layer_name: str, shape: tuple):
        """Initialize parameters for a layer"""
        self.parameters[layer_name] = np.random.normal(0, 0.01, shape).astype(np.float32)
        self.gradients[layer_name] = []
        self.versions[layer_name] = 0
        logger.info(f"Initialized layer {layer_name} with shape {shape}")
    
    def push_gradient(self, layer_name: str, gradient: np.ndarray, worker_id: str) -> int:
        """Store gradient from a worker"""
        if layer_name not in self.gradients:
            self.initialize_layer(layer_name, gradient.shape)
        
        self.gradients[layer_name].append(gradient)
        logger.debug(f"Received gradient for {layer_name} from {worker_id}")
        
        return self.versions[layer_name]
    
    def aggregate_and_update(self, layer_name: str, num_workers: int = 2):
        """Aggregate gradients and update parameters"""
        if layer_name not in self.gradients or len(self.gradients[layer_name]) < num_workers:
            return False  # Not enough gradients yet
        
        # Average all gradients
        avg_gradient = np.mean(self.gradients[layer_name], axis=0)
        
        # Update parameters (simple gradient descent)
        learning_rate = 0.01
        self.parameters[layer_name] -= learning_rate * avg_gradient
        
        # Clear gradients and increment version
        self.gradients[layer_name] = []
        self.versions[layer_name] += 1
        
        logger.info(f"Updated {layer_name} to version {self.versions[layer_name]}")
        return True
    
    def get_parameters(self, layer_name: str) -> tuple:
        """Get current parameters for a layer"""
        if layer_name in self.parameters:
            return self.parameters[layer_name], self.versions[layer_name]
        return None, 0

class ParameterServerServicer(parameter_server_pb2_grpc.ParameterServerServiceServicer):
    """gRPC service implementation - using YOUR proto message names"""
    
    def __init__(self, storage: ParameterStorage):
        self.storage = storage
        self.registered_workers = set()
    
    def RegisterWorker(self, request, context):
        """Register worker - using YOUR message names"""
        worker_id = request.worker_id
        self.registered_workers.add(worker_id)
        logger.info(f"Registered worker {worker_id} with parameter server")
        
        return parameter_server_pb2.ServerRegistrationResponse(
            success=True,
            available_layers=["linear1", "linear2"]  # Available model layers
        )
    
    def PushGradients(self, request, context):
        """Handle gradient push - using YOUR GradientUpdate message"""
        try:
            # Deserialize gradient data
            gradient = pickle.loads(request.gradient_data)
            
            # Store gradient
            version = self.storage.push_gradient(
                request.layer_name, 
                gradient, 
                request.worker_id
            )
            
            # Try to aggregate (if enough gradients collected)
            updated = self.storage.aggregate_and_update(request.layer_name, num_workers=2)
            
            new_version = self.storage.versions[request.layer_name]
            
            return parameter_server_pb2.PushResponse(
                success=True,
                new_version=new_version
            )
            
        except Exception as e:
            logger.error(f"Error pushing gradients: {e}")
            return parameter_server_pb2.PushResponse(
                success=False,
                new_version=0
            )
    
    def PullParameters(self, request, context):
        """Handle parameter pull - using YOUR PullRequest/ParameterUpdate messages"""
        try:
            parameters, version = self.storage.get_parameters(request.layer_name)
            
            if parameters is not None:
                # Serialize parameters
                param_data = pickle.dumps(parameters)
                
                return parameter_server_pb2.ParameterUpdate(
                    layer_name=request.layer_name,
                    parameter_data=param_data,
                    version=version,
                    has_update=True
                )
            else:
                return parameter_server_pb2.ParameterUpdate(
                    layer_name=request.layer_name,
                    parameter_data=b"",
                    version=0,
                    has_update=False
                )
                
        except Exception as e:
            logger.error(f"Error pulling parameters: {e}")
            return parameter_server_pb2.ParameterUpdate(
                layer_name=request.layer_name,
                parameter_data=b"",
                version=0,
                has_update=False
            )

class ParameterServer:
    """Main parameter server class"""
    
    def __init__(self, config: ConfigManager):
        self.config = config.parameter_server
        self.storage = ParameterStorage()
        self.server = None
        self.running = False
    
    async def start(self):
        """Start parameter server"""
        self.running = True
        logger.info(f"Starting Parameter Server on {self.config.host}:{self.config.port}")
        
        # Create gRPC server
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Add service
        parameter_server_pb2_grpc.add_ParameterServerServiceServicer_to_server(
            ParameterServerServicer(self.storage), self.server
        )
        
        # Start server
        listen_addr = f"{self.config.host}:{self.config.port}"
        self.server.add_insecure_port(listen_addr)
        await self.server.start()
        
        logger.info(f"Parameter Server listening on {listen_addr}")
        
        # Keep running
        try:
            await self.server.wait_for_termination()
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        """Stop server"""
        self.running = False
        if self.server:
            await self.server.stop(grace=5)