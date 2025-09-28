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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Complete HPO example with multiple trials"""
    config = ConfigManager()
    
    logger.info("🔬 Starting Complete HPO Example - Multiple Trials")
    
    # Start services
    coordinator = GRPCCoordinator(config)
    param_server = ParameterServer(config)
    
    master_task = asyncio.create_task(coordinator.start())
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(3)
    
    try:
        # Create HPO service
        hpo_service = HPOService(coordinator, config)
        
        # Full search space with multiple parameters
        search_space = {
            'learning_rate': {
                'type': 'float',
                'min': 0.001,
                'max': 0.01
            },
            'batch_size': {
                'type': 'int', 
                'min': 16,
                'max': 64
            }
        }
        
        # Create HPO experiment with multiple trials
        hpo_config = HPOConfig(
            experiment_name="complete_mnist_hpo",
            search_space=search_space,
            objective_metric="accuracy",
            optimization_direction="maximize",
            max_trials=5,
            max_parallel_trials=1  # Sequential for clear logging
        )
        
        experiment_id = await hpo_service.create_experiment(hpo_config)
        logger.info(f"🧪 Started HPO experiment: {experiment_id}")
        
        # Monitor experiment progress with detailed logging
        print("Monitoring HPO experiment - this will run 5 trials...")

        # Give the experiment loop time to start properly
        await asyncio.sleep(10)  # Increased from 5 to 10 seconds

        max_checks = 200  # Increased from 150 to 200
        check_interval = 15  # Keep at 15 seconds

        for i in range(max_checks):
            await asyncio.sleep(check_interval)
            
            try:
                status = await hpo_service.get_experiment_status(experiment_id)
            except Exception as e:
                print(f"Error getting status: {e}")
                continue
            
            completed = status.get('completed_trials', 0)
            failed = status.get('failed_trials', 0)
            running = status.get('running_trials', 0)
            total = status.get('total_trials', 0)
            exp_status = status.get('status', 'unknown')
            
            print(f"Check {i+1}: Status={exp_status}, Progress={completed+failed}/{total} trials "
                f"(Completed: {completed}, Failed: {failed}, Running: {running})")
            
            if best_trial := status.get('best_trial'):
                best_metrics = best_trial.get('metrics', {})
                best_params = best_trial.get('parameters', {})
                print(f"  Current best: accuracy={best_metrics.get('accuracy', 0):.2f}%, params={best_params}")
            
            # WAIT FOR ALL TRIALS - be more patient
            if exp_status == 'completed' and completed + failed >= total and total > 0:
                print("🎉 HPO experiment completed!")
                if best_trial:
                    print(f"🏆 Best parameters: {best_trial.get('parameters')}")
                    print(f"🏆 Best metrics: {best_trial.get('metrics')}")
                break
            elif exp_status == 'failed':
                print("❌ HPO experiment failed!")
                break
            
            # Show patience messages
            if i > 0 and i % 4 == 0:  # Every minute
                print(f"  Waiting patiently... {completed}/{total} trials completed, monitoring continues...")

        print("Monitoring finished.")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        logger.info("🛑 Shutting down...")
        await param_server.stop()
        await coordinator.stop()
        ps_task.cancel()
        master_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())