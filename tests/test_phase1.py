import sys
import os
import asyncio
import logging
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.master.grpc_coordinator import GRPCCoordinator
from src.worker.grpc_worker import GRPCWorker
from src.utils.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_worker_registration():
    """Test: Workers can register with master"""
    config = ConfigManager()
    coordinator = GRPCCoordinator(config)
    
    # Start master
    master_task = asyncio.create_task(coordinator.start())
    await asyncio.sleep(1)
    
    # Start worker
    worker = GRPCWorker("test_worker_reg", config)
    worker_task = asyncio.create_task(worker.start())
    await asyncio.sleep(2)
    
    # Check registration
    assert "test_worker_reg" in coordinator.workers, "Worker not registered"
    print("✓ Worker registration working")
    
    # Cleanup
    await worker.stop()
    await coordinator.stop()
    worker_task.cancel()
    master_task.cancel()

async def test_heartbeat_system():
    """Test: Heartbeat system detects failures"""
    config = ConfigManager()
    coordinator = GRPCCoordinator(config)
    
    master_task = asyncio.create_task(coordinator.start())
    await asyncio.sleep(1)
    
    worker = GRPCWorker("test_worker_hb", config)
    worker_task = asyncio.create_task(worker.start())
    await asyncio.sleep(2)
    
    # Verify heartbeat is working
    initial_time = coordinator.workers["test_worker_hb"]["last_heartbeat"]
    await asyncio.sleep(35)  # Wait for heartbeat
    new_time = coordinator.workers["test_worker_hb"]["last_heartbeat"]
    
    assert new_time > initial_time, "Heartbeat not updating"
    print("✓ Heartbeat system working")
    
    # Cleanup
    await worker.stop()
    await coordinator.stop()
    worker_task.cancel()
    master_task.cancel()

async def test_multiple_workers():
    """Test: Multiple workers can connect simultaneously"""
    config = ConfigManager()
    coordinator = GRPCCoordinator(config)
    
    master_task = asyncio.create_task(coordinator.start())
    await asyncio.sleep(1)
    
    # Start 3 workers
    workers = []
    worker_tasks = []
    for i in range(3):
        worker = GRPCWorker(f"multi_worker_{i}", config)
        worker_task = asyncio.create_task(worker.start())
        workers.append(worker)
        worker_tasks.append(worker_task)
    
    await asyncio.sleep(3)
    
    # Check all registered
    assert len(coordinator.workers) == 3, f"Expected 3 workers, got {len(coordinator.workers)}"
    print("✓ Multiple workers can connect")
    
    # Cleanup
    for worker in workers:
        await worker.stop()
    await coordinator.stop()
    for task in worker_tasks:
        task.cancel()
    master_task.cancel()

async def test_failure_detection():
    """Test: Master detects worker failures"""
    config = ConfigManager()
    coordinator = GRPCCoordinator(config)
    
    master_task = asyncio.create_task(coordinator.start())
    await asyncio.sleep(1)
    
    worker = GRPCWorker("failure_test", config)
    worker_task = asyncio.create_task(worker.start())
    await asyncio.sleep(2)
    
    # Simulate worker failure by stopping without cleanup
    worker.running = False
    worker_task.cancel()
    
    # Wait for failure detection (should timeout after 60 seconds)
    await asyncio.sleep(95)
    
    assert "failure_test" not in coordinator.workers, "Failed worker not removed"
    print("✓ Failure detection working")
    
    await coordinator.stop()
    master_task.cancel()

async def run_all_tests():
    """Run all Phase 1 tests"""
    print("Starting Phase 1 Verification Tests...\n")
    
    try:
        await test_worker_registration()
        await test_heartbeat_system()
        await test_multiple_workers()
        await test_failure_detection()
        
        print("\n🎉 ALL PHASE 1 TESTS PASSED!")
        print("Phase 1 is fully functional and ready for Phase 2")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all_tests())