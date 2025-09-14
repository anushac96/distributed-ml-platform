import asyncio
import logging
from typing import Dict, List
import grpc
from concurrent import futures
import time

from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

class MasterCoordinator:
    def __init__(self, config: ConfigManager):
        self.config = config.master
        self.workers: Dict[str, dict] = {}  # worker_id -> worker_info
        self.jobs: Dict[str, dict] = {}     # job_id -> job_info
        self.running = False
        
    async def start(self):
        """Start the master coordinator service"""
        self.running = True
        logger.info(f"Starting Master Coordinator on {self.config.host}:{self.config.port}")
        
        # Start heartbeat monitoring
        asyncio.create_task(self.monitor_workers())
        
        # Start gRPC server (placeholder for now)
        await self._start_grpc_server()
    
    async def register_worker(self, worker_id: str, worker_info: dict):
        """Register a new worker node"""
        self.workers[worker_id] = {
            **worker_info,
            'status': 'active',
            'last_heartbeat': time.time(),
            'jobs': []
        }
        logger.info(f"Registered worker: {worker_id}")
    
    async def monitor_workers(self):
        """Monitor worker heartbeats and handle failures"""
        while self.running:
            current_time = time.time()
            dead_workers = []
            
            for worker_id, worker_info in self.workers.items():
                if current_time - worker_info['last_heartbeat'] > self.config.heartbeat_interval * 2:
                    dead_workers.append(worker_id)
            
            for worker_id in dead_workers:
                await self.handle_worker_failure(worker_id)
            
            await asyncio.sleep(self.config.heartbeat_interval)
    
    async def handle_worker_failure(self, worker_id: str):
        """Handle worker node failure"""
        logger.warning(f"Worker {worker_id} failed - reassigning jobs")
        worker = self.workers.pop(worker_id, None)
        if worker and worker['jobs']:
            # Reassign jobs to other workers
            for job_id in worker['jobs']:
                await self.reassign_job(job_id)
    
    async def reassign_job(self, job_id: str):
        """Reassign job to available worker"""
        # Implementation for job reassignment
        logger.info(f"Reassigning job {job_id}")
    
    async def _start_grpc_server(self):
        """Placeholder for gRPC server"""
        # Will implement actual gRPC server later
        while self.running:
            await asyncio.sleep(1)