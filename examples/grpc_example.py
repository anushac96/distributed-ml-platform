import sys
import os
import asyncio
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.master.grpc_coordinator import GRPCCoordinator
from src.worker.grpc_worker import GRPCWorker
from src.utils.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Test full gRPC system"""
    config = ConfigManager()
    
    # Start master coordinator
    coordinator = GRPCCoordinator(config)
    master_task = asyncio.create_task(coordinator.start())
    
    # Wait for master to start
    await asyncio.sleep(2)
    
    # Start workers
    workers = []
    worker_tasks = []
    
    for i in range(2):
        worker = GRPCWorker(f"worker_{i}", config)
        worker_task = asyncio.create_task(worker.start())
        workers.append(worker)
        worker_tasks.append(worker_task)
    
    logger.info("Started full gRPC distributed system")
    logger.info("Watch for worker registration and heartbeat messages")
    
    # Run for 60 seconds
    await asyncio.sleep(60)
    
    # Cleanup
    logger.info("Shutting down...")
    for worker in workers:
        await worker.stop()
    await coordinator.stop()
    
    for task in worker_tasks:
        task.cancel()
    master_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")