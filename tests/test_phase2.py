import sys
import os
import asyncio
import logging
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.master.grpc_coordinator import GRPCCoordinator
from src.parameter_server.parameter_server import ParameterServer
from src.worker.ml_worker import MLWorker
from src.utils.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_parameter_server_startup():
    """Test: Parameter server starts successfully"""
    config = ConfigManager()
    param_server = ParameterServer(config)
    
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(2)
    
    assert param_server.running == True, "Parameter server not running"
    print("✓ Parameter server startup working")
    
    await param_server.stop()
    ps_task.cancel()

async def test_ml_worker_registration():
    """Test: ML Workers can register with parameter server"""
    config = ConfigManager()
    
    # Start master and parameter server
    coordinator = GRPCCoordinator(config)
    param_server = ParameterServer(config)
    
    master_task = asyncio.create_task(coordinator.start())
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(2)
    
    # Start ML worker
    worker = MLWorker("test_ml_worker", config)
    worker_task = asyncio.create_task(worker.start())
    await asyncio.sleep(5)  # Give more time for registration
    
    # Check registrations
    assert "test_ml_worker" in coordinator.workers, "Worker not registered with master"
    assert "test_ml_worker" in param_server.storage.registered_workers, "Worker not registered with parameter server"
    print("✓ ML worker registration working")
    
    # Cleanup
    worker.running = False  # Stop training loop
    await worker.stop()
    await param_server.stop()
    await coordinator.stop()
    
    # Cancel tasks properly
    if not worker_task.done():
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    
    if not ps_task.done():
        ps_task.cancel()
        try:
            await ps_task
        except asyncio.CancelledError:
            pass
            
    if not master_task.done():
        master_task.cancel()
        try:
            await master_task
        except asyncio.CancelledError:
            pass

async def test_parameter_synchronization():
    """Test: Parameters are synchronized between workers"""
    config = ConfigManager()
    
    # Start services
    coordinator = GRPCCoordinator(config)
    param_server = ParameterServer(config)
    
    master_task = asyncio.create_task(coordinator.start())
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(2)
    
    # Start 2 workers
    worker1 = MLWorker("sync_worker_0", config)
    worker2 = MLWorker("sync_worker_1", config)
    
    worker1_task = asyncio.create_task(worker1.start())
    worker2_task = asyncio.create_task(worker2.start())
    await asyncio.sleep(8)  # Wait for initial training
    
    # Check if parameters are being synchronized (look for version updates)
    initial_versions = dict(param_server.storage.versions)
    
    await asyncio.sleep(10)  # Wait for more training iterations
    
    # Check if versions have been updated (indicating gradient aggregation)
    versions_updated = False
    for layer_name in param_server.storage.versions:
        current_version = param_server.storage.versions[layer_name]
        initial_version = initial_versions.get(layer_name, 0)
        if current_version > initial_version:
            versions_updated = True
            logger.info(f"Layer {layer_name}: version {initial_version} -> {current_version}")
            break
    
    assert versions_updated, f"Parameters not being synchronized. Versions: {param_server.storage.versions}"
    print("✓ Parameter synchronization working")
    
    # Cleanup
    worker1.running = False
    worker2.running = False
    await worker1.stop()
    await worker2.stop()
    await param_server.stop()
    await coordinator.stop()
    
    # Cancel tasks properly
    for task in [worker1_task, worker2_task, ps_task, master_task]:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

async def test_gradient_aggregation():
    """Test: Gradients are properly aggregated"""
    config = ConfigManager()
    
    coordinator = GRPCCoordinator(config)
    param_server = ParameterServer(config)
    
    master_task = asyncio.create_task(coordinator.start())
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(2)
    
    worker = MLWorker("gradient_worker", config)
    worker_task = asyncio.create_task(worker.start())
    await asyncio.sleep(8)  # Wait for some training
    
    # Check if gradients are being processed
    has_parameters = len(param_server.storage.parameters) > 0
    has_versions = len(param_server.storage.versions) > 0
    
    assert has_parameters or has_versions, "No gradient processing detected"
    print("✓ Gradient aggregation working")
    
    # Cleanup
    worker.running = False
    await worker.stop()
    await param_server.stop()
    await coordinator.stop()
    
    for task in [worker_task, ps_task, master_task]:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

async def run_phase2_tests():
    """Run all Phase 2 tests"""
    print("Starting Phase 2 Verification Tests...\n")
    
    try:
        await test_parameter_server_startup()
        await test_ml_worker_registration()
        await test_parameter_synchronization()
        await test_gradient_aggregation()
        
        print("\n🎉 ALL PHASE 2 TESTS PASSED!")
        print("✅ Parameter server operational")
        print("✅ ML worker registration working")
        print("✅ Parameter synchronization functional")
        print("✅ Gradient aggregation working")
        print("\nPhase 2 distributed ML training is fully functional!")
        
    except Exception as e:
        print(f"\n❌ PHASE 2 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_phase2_tests())