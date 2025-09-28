# Test: HPO + Mock Services
# test with mock coordinator to isolate HPO logic
import sys
import os
import asyncio
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.hpo.hpo_service import HPOService, HPOConfig
from src.utils.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockCoordinator:
    def __init__(self):
        self.workers = {}

async def main():
    print("Testing HPO with mock services...")
    
    config = ConfigManager()
    coordinator = MockCoordinator()
    hpo_service = HPOService(coordinator, config)
    
    print("Creating experiment...")
    hpo_config = HPOConfig(
        experiment_name="isolated_test",
        search_space={'learning_rate': {'type': 'float', 'min': 0.001, 'max': 0.01}},
        objective_metric="accuracy",
        max_trials=1,
        max_parallel_trials=1
    )
    
    # Don't actually create the experiment, just test the setup
    print("HPO config created successfully")
    
    # Test Bayesian optimizer directly
    optimizer = hpo_service.experiments.get('test', {}).get('optimizer')
    if not optimizer:
        from src.hpo.bayesian_optimizer import BayesianOptimizer
        optimizer = BayesianOptimizer(hpo_config.search_space, "accuracy", "maximize")
    
    suggestion = await optimizer.suggest()
    print(f"Parameter suggestion: {suggestion}")
    
    print("SUCCESS: HPO isolated test working")

if __name__ == "__main__":
    asyncio.run(main())