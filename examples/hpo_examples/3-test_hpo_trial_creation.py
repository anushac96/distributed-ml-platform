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

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    print("🧪 Testing HPO with multiple trials - DEBUG VERSION")
    
    config = ConfigManager()
    
    # Start services
    print("Starting gRPC services...")
    coordinator = GRPCCoordinator(config)
    param_server = ParameterServer(config)
    
    master_task = asyncio.create_task(coordinator.start())
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(3)  # Give services time to start
    print("✅ Services started")
    
    try:
        # Create HPO service
        hpo_service = HPOService(coordinator, config)
        
        # Create experiment - start with 3 trials
        print("\n🔬 Creating HPO experiment...")
        hpo_config = HPOConfig(
            experiment_name="debug_test",
            search_space={'learning_rate': {'type': 'float', 'min': 0.001, 'max': 0.01}},
            objective_metric="accuracy",
            optimization_direction="maximize",
            max_trials=3,
            max_parallel_trials=1  # Sequential for clear debugging
        )
        
        experiment_id = await hpo_service.create_experiment(hpo_config)
        print(f"📊 Experiment created: {experiment_id}")
        print(f"Target: {hpo_config.max_trials} trials\n")
        
        # Enhanced monitoring with more frequent checks
        print("👀 Starting detailed monitoring...")
        max_checks = 60  # 10 minutes of monitoring
        check_interval = 10  # Check every 10 seconds
        
        for check_num in range(1, max_checks + 1):
            await asyncio.sleep(check_interval)
            
            # Get detailed status
            status = await hpo_service.get_experiment_status(experiment_id)
            
            completed = status.get('completed_trials', 0)
            failed = status.get('failed_trials', 0)
            running = status.get('running_trials', 0)
            total_target = status.get('total_trials', 0)
            exp_status = status.get('status', 'unknown')
            
            # Progress indicators
            progress_bar = "█" * completed + "░" * (total_target - completed - failed - running) + "🔄" * running
            
            print(f"📈 Check {check_num:2d}: [{progress_bar}] "
                  f"Status: {exp_status} | "
                  f"✅{completed} ❌{failed} 🔄{running} 🎯{total_target}")
            
            # Show best trial info
            best_trial = status.get('best_trial')
            if best_trial:
                best_acc = best_trial.get('metrics', {}).get('accuracy', 0)
                best_lr = best_trial.get('parameters', {}).get('learning_rate', 0)
                print(f"    🏆 Best so far: accuracy={best_acc:.2f}%, lr={best_lr:.4f}")
            
            # Check if experiment is done
            total_finished = completed + failed
            if exp_status == 'completed':
                print(f"\n🎉 SUCCESS! Experiment completed with {completed} successful trials!")
                if best_trial:
                    print(f"🏆 Final best: {best_trial}")
                break
            elif exp_status == 'failed':
                print(f"\n❌ FAILED! Experiment failed")
                break
            elif total_finished >= total_target:
                print(f"\n✅ All trials finished: {completed} completed, {failed} failed")
                break
            elif check_num % 6 == 0:  # Every minute, show more detail
                print(f"    ℹ️  Active trials in HPO service: {len(hpo_service.active_trials)}")
                print(f"    ℹ️  Trials in experiment: {len(hpo_service.experiments[experiment_id]['trials'])}")
                
        else:
            print(f"\n⏰ Monitoring timeout after {max_checks * check_interval} seconds")
            final_status = await hpo_service.get_experiment_status(experiment_id)
            print(f"Final status: {final_status}")

        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"\n💥 ERROR during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🛑 Shutting down services...")
        try:
            await param_server.stop()
            await coordinator.stop()
            ps_task.cancel()
            master_task.cancel()
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")
        print("✅ Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())