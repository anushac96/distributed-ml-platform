import sys
import os
import asyncio
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.master.grpc_coordinator import GRPCCoordinator
from src.parameter_server.parameter_server import ParameterServer
from src.worker.ml_worker import MLWorker
from src.utils.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Test distributed ML training system"""
    config = ConfigManager()
    
    logger.info("🚀 Starting Distributed ML Training System")
    
    # Start master coordinator
    coordinator = GRPCCoordinator(config)
    master_task = asyncio.create_task(coordinator.start())
    await asyncio.sleep(1)
    
    # Start parameter server  
    param_server = ParameterServer(config)
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(1)
    
    # Start ML workers
    workers = []
    worker_tasks = []
    
    for i in range(2):
        worker = MLWorker(f"ml_worker_{i}", config)
        worker_task = asyncio.create_task(worker.start())
        workers.append(worker)
        worker_tasks.append(worker_task)
    
    logger.info("📊 System Status:")
    logger.info("   Master Coordinator: localhost:50051") 
    logger.info("   Parameter Server: localhost:50052")
    logger.info("   ML Workers: 2 training nodes")
    logger.info("   Model: Simple Neural Network (784->128->10)")
    
    # Let training run for 35 seconds (should complete ~10 iterations)
    await asyncio.sleep(35)
    
    # Cleanup
    logger.info("🛑 Shutting down distributed training system...")
    for worker in workers:
        await worker.stop()
    await param_server.stop()
    await coordinator.stop()
    
    for task in worker_tasks:
        if not task.done():
            task.cancel()
    if not ps_task.done():
        ps_task.cancel()
    if not master_task.done():
        master_task.cancel()
    
    logger.info("✅ Distributed training system shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Training interrupted by user")