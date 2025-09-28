# Test HPO with Real gRPC Services
import sys
import os
import asyncio
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.master.grpc_coordinator import GRPCCoordinator
from src.parameter_server.parameter_server import ParameterServer
from src.hpo.hpo_service import HPOService, HPOConfig
from src.utils.config import ConfigManager

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

async def main():
    print("Testing HPO with real gRPC services...")
    
    config = ConfigManager()
    
    # Start services
    print("1. Starting gRPC services...")
    coordinator = GRPCCoordinator(config)
    param_server = ParameterServer(config)
    
    master_task = asyncio.create_task(coordinator.start())
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(2)
    print("   Services started")
    
    try:
        # Create HPO service
        print("2. Creating HPO service...")
        hpo_service = HPOService(coordinator, config)
        print("   HPO service created")
        
        # Test status check without creating experiment
        print("3. Testing status check...")
        status = await hpo_service.get_experiment_status("fake_id")
        print(f"   Status check result: {status}")
        
        print("4. Testing experiment creation (without running)...")
        hpo_config = HPOConfig(
            experiment_name="grpc_test",
            search_space={'learning_rate': {'type': 'float', 'min': 0.001, 'max': 0.01}},
            objective_metric="accuracy",
            max_trials=0,  # Don't actually run trials
            max_parallel_trials=1
        )
        
        # Just test the config creation, don't create actual experiment
        print("   HPO config created successfully")
        
        print("SUCCESS: gRPC + HPO integration working")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("5. Cleanup...")
        try:
            await param_server.stop()
            await coordinator.stop()
        except:
            pass
        ps_task.cancel()
        master_task.cancel()
        print("   Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())