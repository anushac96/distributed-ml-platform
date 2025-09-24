import sys
import os
import asyncio
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.master.grpc_coordinator import GRPCCoordinator
from src.parameter_server.parameter_server import ParameterServer
from src.worker.production_ml_worker import ProductionMLWorker
from src.utils.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Production distributed ML training with real datasets"""
    config = ConfigManager()
    
    logger.info("🏭 Starting Production Distributed ML Training System")
    
    # Start master coordinator
    coordinator = GRPCCoordinator(config)
    master_task = asyncio.create_task(coordinator.start())
    await asyncio.sleep(1)
    
    # Start parameter server
    param_server = ParameterServer(config)
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(1)
    
    # Start production ML workers
    workers = []
    worker_tasks = []
    
    # Choose dataset and model
    dataset = "mnist"      # or "cifar10"
    model_name = "cnn"     # or "resnet18" for CIFAR-10
    
    for i in range(2):
        worker = ProductionMLWorker(
            f"ml_worker_{i}", 
            config, 
            dataset=dataset,
            model_name=model_name
        )
        worker_task = asyncio.create_task(worker.start())
        workers.append(worker)
        worker_tasks.append(worker_task)
    
    logger.info("📊 Production System Status:")
    logger.info(f"   Master Coordinator: localhost:50051")
    logger.info(f"   Parameter Server: localhost:50052")
    logger.info(f"   ML Workers: 2 production training nodes")
    logger.info(f"   Dataset: {dataset.upper()}")
    logger.info(f"   Model: {model_name.upper()}")
    logger.info(f"   Features: Real data, checkpointing, distributed loading")
    
    # Let training run for full epochs (will take longer with real data)
    logger.info("🚀 Training started - this will take several minutes with real data...")
    await asyncio.sleep(300)  # 5 minutes for real training
    
    # Cleanup
    logger.info("🛑 Shutting down production training system...")
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
    
    logger.info("✅ Production training system shutdown complete")
    logger.info("📁 Check ./checkpoints/ directory for saved models")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Production training interrupted by user")