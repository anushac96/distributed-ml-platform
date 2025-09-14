import asyncio
import logging
from src.master.coordinator import MasterCoordinator
from src.worker.worker import WorkerNode
from src.utils.config import ConfigManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_basic_example():
    """Run a basic distributed training example"""
    config = ConfigManager()
    
    # Start master coordinator
    master = MasterCoordinator(config)
    master_task = asyncio.create_task(master.start())
    
    # Wait a bit for master to start
    await asyncio.sleep(1)
    
    # Start worker nodes
    workers = []
    for i in range(2):  # Start 2 workers
        worker = WorkerNode(f"worker_{i}", config)
        worker_task = asyncio.create_task(worker.start())
        workers.append(worker_task)
    
    logger.info("Started distributed training system")
    logger.info("Master: localhost:50051")
    logger.info("Workers: 2 nodes")
    
    # Run for a short time then shutdown
    await asyncio.sleep(10)
    
    # Cleanup
    logger.info("Shutting down...")
    master.running = False
    for worker_task in workers:
        worker_task.cancel()
    
    master_task.cancel()
    logger.info("Shutdown complete")

if __name__ == "__main__":
    asyncio.run(run_basic_example())