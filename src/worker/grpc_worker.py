import asyncio
import logging
import grpc
import time

from ..generated import master_pb2
from ..generated import master_pb2_grpc
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

class GRPCWorker:
    def __init__(self, worker_id: str, config: ConfigManager):
        self.worker_id = worker_id
        self.config = config.worker
        self.running = False
        self.channel = None
        self.stub = None
        
    async def start(self):
        """Start worker and connect to master"""
        self.running = True
        logger.info(f"Starting gRPC Worker: {self.worker_id}")
        
        # Connect to master
        await self.connect_to_master()
        
        # Register with master
        await self.register()
        
        # Start heartbeat
        asyncio.create_task(self.heartbeat_loop())
        
        # Keep running
        while self.running:
            await asyncio.sleep(1)
    
    async def connect_to_master(self):
        """Connect to master via gRPC"""
        master_addr = f"{self.config.master_host}:{self.config.master_port}"
        self.channel = grpc.aio.insecure_channel(master_addr)
        self.stub = master_pb2_grpc.MasterServiceStub(self.channel)
        logger.info(f"Connected to master at {master_addr}")
    
    async def register(self):
        """Register with master"""
        try:
            capabilities = master_pb2.WorkerCapabilities(
                gpu_enabled=self.config.gpu_enabled,
                max_batch_size=self.config.max_batch_size,
                device_type="cpu",  # or "cuda" if GPU
                supported_models=["test_model"]
            )
            
            request = master_pb2.RegisterWorkerRequest(
                worker_id=self.worker_id,
                capabilities=capabilities,
                host="localhost",
                port=0
            )
            
            response = await self.stub.RegisterWorker(request)
            
            if response.success:
                logger.info(f"Registration successful: {response.message}")
            else:
                logger.error(f"Registration failed: {response.message}")
                
        except grpc.RpcError as e:
            logger.error(f"Registration RPC failed: {e}")
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                status = master_pb2.WorkerStatus(
                    current_job_id="",
                    cpu_usage=0.1,
                    memory_usage=0.2,
                    gpu_usage=0.0,
                    completed_batches=0
                )
                
                request = master_pb2.HeartbeatRequest(
                    worker_id=self.worker_id,
                    status=status,
                    timestamp=int(time.time())
                )
                
                response = await self.stub.SendHeartbeat(request)
                
                if response.acknowledged:
                    logger.debug(f"Heartbeat acknowledged")
                else:
                    logger.warning("Heartbeat not acknowledged")
                    
            except grpc.RpcError as e:
                logger.error(f"Heartbeat failed: {e}")
            
            await asyncio.sleep(30)
    
    async def stop(self):
        """Stop worker"""
        self.running = False
        if self.channel:
            await self.channel.close()