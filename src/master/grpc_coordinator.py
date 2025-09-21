import asyncio
import logging
from typing import Dict
import grpc
from concurrent import futures
import time

# Import generated gRPC code
from ..generated import master_pb2
from ..generated import master_pb2_grpc
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

class MasterServicer(master_pb2_grpc.MasterServiceServicer):
    """gRPC service implementation"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
    
    def RegisterWorker(self, request, context):
        """Handle worker registration"""
        try:
            worker_info = {
                'host': request.host,
                'port': request.port,
                'capabilities': {
                    'gpu_enabled': request.capabilities.gpu_enabled,
                    'max_batch_size': request.capabilities.max_batch_size,
                    'device_type': request.capabilities.device_type,
                }
            }
            
            # Register synchronously - NO asyncio.create_task!
            self.coordinator.workers[request.worker_id] = worker_info
            logger.info(f"Registered worker: {request.worker_id}")
            
            return master_pb2.RegisterWorkerResponse(
                success=True,
                message=f"Worker {request.worker_id} registered",
                assigned_worker_id=request.worker_id
            )
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return master_pb2.RegisterWorkerResponse(
                success=False,
                message=str(e),
                assigned_worker_id=""
            )
    
    def SendHeartbeat(self, request, context):
        """Handle heartbeat"""
        worker_id = request.worker_id
        
        if worker_id in self.coordinator.workers:
            self.coordinator.workers[worker_id]['last_heartbeat'] = time.time()
            return master_pb2.HeartbeatResponse(acknowledged=True)
        else:
            return master_pb2.HeartbeatResponse(acknowledged=False)

class GRPCCoordinator:
    def __init__(self, config: ConfigManager):
        self.config = config.master
        self.workers: Dict[str, dict] = {}
        self.running = False
        self.server = None
        
    async def start(self):
        """Start gRPC server and coordinator"""
        self.running = True
        logger.info(f"Starting gRPC Coordinator on {self.config.host}:{self.config.port}")
        
        # Create gRPC server
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Add our service
        master_pb2_grpc.add_MasterServiceServicer_to_server(
            MasterServicer(self), self.server
        )
        
        # Start server
        listen_addr = f"{self.config.host}:{self.config.port}"
        self.server.add_insecure_port(listen_addr)
        
        await self.server.start()
        logger.info(f"gRPC server listening on {listen_addr}")
        
        # Start background tasks
        asyncio.create_task(self.monitor_workers())
        
        # Keep server running
        try:
            await self.server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await self.stop()
    
    async def register_worker(self, worker_id: str, worker_info: dict):
        """Register worker"""
        self.workers[worker_id] = {
            **worker_info,
            'status': 'active',
            'last_heartbeat': time.time(),
        }
        logger.info(f"Registered worker: {worker_id}")
    
    async def monitor_workers(self):
        """Monitor worker health"""
        while self.running:
            current_time = time.time()
            dead_workers = []
            
            for worker_id, worker_info in self.workers.items():
                time_since_heartbeat = current_time - worker_info['last_heartbeat']
                if time_since_heartbeat > 60:  # 60 second timeout
                    dead_workers.append(worker_id)
            
            for worker_id in dead_workers:
                logger.warning(f"Worker {worker_id} timed out, removing")
                self.workers.pop(worker_id, None)
            
            if self.workers:
                logger.info(f"Active workers: {list(self.workers.keys())}")
            
            await asyncio.sleep(30)
    
    async def stop(self):
        """Stop the server"""
        self.running = False
        if self.server:
            await self.server.stop(grace=5)