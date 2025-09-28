import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    DATA_VALIDATION = "data_validation"
    PREPROCESSING = "preprocessing"  
    HPO = "hyperparameter_optimization"
    TRAINING = "training"
    AB_TESTING = "ab_testing"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"

@dataclass
class PipelineConfig:
    name: str
    stages: List[PipelineStage]
    dataset: str = "mnist"
    model: str = "cnn"

class PipelineManager:
    """Orchestrate end-to-end ML pipelines"""
    
    def __init__(self, coordinator, hpo_service, ab_test_manager):
        self.coordinator = coordinator
        self.hpo_service = hpo_service
        self.ab_test_manager = ab_test_manager
        self.pipelines: Dict[str, Dict] = {}
    
    async def create_pipeline(self, config: PipelineConfig) -> str:
        """Create and start ML pipeline"""
        pipeline_id = f"pipeline_{config.name}"
        
        pipeline = {
            'id': pipeline_id,
            'config': config,
            'status': 'running',
            'current_stage': 0,
            'results': {}
        }
        
        self.pipelines[pipeline_id] = pipeline
        
        # Start pipeline execution
        asyncio.create_task(self._execute_pipeline(pipeline_id))
        
        logger.info(f"Started pipeline: {config.name}")
        return pipeline_id
    
    async def _execute_pipeline(self, pipeline_id: str):
        """Execute pipeline stages sequentially"""
        pipeline = self.pipelines[pipeline_id]
        config = pipeline['config']
        
        for i, stage in enumerate(config.stages):
            pipeline['current_stage'] = i
            logger.info(f"Executing stage: {stage.value}")
            
            try:
                result = await self._execute_stage(stage, config)
                pipeline['results'][stage.value] = result
                
                # Log successful stage completion
                if stage == PipelineStage.HPO:
                    completed_trials = result.get('completed_trials', 0)
                    logger.info(f"HPO stage completed with {completed_trials} trials")
                
            except Exception as e:
                logger.error(f"Pipeline stage {stage.value} failed: {e}")
                pipeline['status'] = 'failed'
                return
        
        pipeline['status'] = 'completed'
        logger.info(f"Pipeline {pipeline_id} completed successfully")
    
    async def _execute_stage(self, stage: PipelineStage, config: PipelineConfig) -> Dict[str, Any]:
        """Execute individual pipeline stage"""
        
        if stage == PipelineStage.DATA_VALIDATION:
            logger.info(f"Validating {config.dataset} dataset")
            await asyncio.sleep(1)  # Simulate validation
            return {"validation_passed": True}
            
        elif stage == PipelineStage.HPO:
            logger.info(f"Starting hyperparameter optimization for {config.dataset}")
            
            # Create HPO configuration
            from ..hpo.hpo_service import HPOConfig
            hpo_config = HPOConfig(
                experiment_name=f"{config.name}_hpo",
                search_space={
                    'learning_rate': {'type': 'float', 'min': 0.001, 'max': 0.01},
                    'batch_size': {'type': 'int', 'min': 16, 'max': 64}
                },
                objective_metric="accuracy",
                optimization_direction="maximize",
                max_trials=3,
                max_parallel_trials=1
            )
            
            # Run HPO experiment
            experiment_id = await self.hpo_service.create_experiment(hpo_config)
            logger.info(f"Created HPO experiment: {experiment_id}")
            
            # CRITICAL: Wait long enough for experiment to complete ALL trials
            total_wait_time = 600  # 10 minutes total
            check_interval = 5     # Check every 5 seconds
            max_iterations = total_wait_time // check_interval
            
            await asyncio.sleep(10)  # Initial setup time
            
            for iteration in range(max_iterations):
                await asyncio.sleep(check_interval)
                
                try:
                    status = await self.hpo_service.get_experiment_status(experiment_id)
                    completed = status.get('completed_trials', 0)
                    failed = status.get('failed_trials', 0)
                    running = status.get('running_trials', 0)
                    total = status.get('total_trials', 0)
                    exp_status = status.get('status', 'unknown')
                    
                    # Log progress every 30 seconds
                    if iteration % 6 == 0:
                        logger.info(f"HPO Progress: {completed+failed}/{total} trials, status: {exp_status}")
                    
                    # Only exit when experiment is truly complete
                    if exp_status == 'completed' and completed + failed >= total:
                        logger.info(f"HPO experiment completed: {completed} successful, {failed} failed")
                        break
                    elif exp_status == 'failed':
                        logger.error("HPO experiment failed")
                        break
                except Exception as e:
                    logger.error(f"Error checking HPO status: {e}")
            
            # Get final results
            final_status = await self.hpo_service.get_experiment_status(experiment_id)
            best_trial = final_status.get('best_trial', {})
            
            if not best_trial:
                best_trial = {'parameters': {'learning_rate': 0.001, 'batch_size': 32}, 'metrics': {'accuracy': 0}}
            
            logger.info(f"HPO completed. Best parameters: {best_trial.get('parameters')}")
            
            return {
                "best_parameters": best_trial.get('parameters', {}),
                "best_metrics": best_trial.get('metrics', {}),
                "experiment_id": experiment_id,
                "completed_trials": final_status.get('completed_trials', 0)
            }

        elif stage == PipelineStage.EVALUATION:
            logger.info("Starting model evaluation")
            
            # Get results from previous stages
            pipeline = self.pipelines.get(f"pipeline_{config.name}", {})
            results = pipeline.get('results', {})
            hpo_results = results.get('hyperparameter_optimization', {})
            ab_results = results.get('ab_testing', {})
            
            # Simulate evaluation metrics
            evaluation_results = {
                "final_accuracy": 85.2,
                "final_loss": 0.234,
                "test_accuracy": 83.7,
                "best_hpo_params": hpo_results.get('best_parameters', {}),
                "ab_test_winner": ab_results.get('winner', 'control'),
                "model_size": "688,138 parameters",
                "training_time": "45.2 seconds"
            }
            
            logger.info(f"Evaluation completed: {evaluation_results['final_accuracy']:.1f}% accuracy")
            return evaluation_results
            
        elif stage == PipelineStage.AB_TESTING:
            logger.info("Starting A/B testing")
            
            # Get HPO results from previous stage
            pipeline = self.pipelines.get(f"pipeline_{config.name}", {})
            hpo_results = pipeline.get('results', {}).get('hyperparameter_optimization', {})
            best_params = hpo_results.get('best_parameters', {'learning_rate': 0.001, 'batch_size': 32})
            
            # Create A/B test configuration
            from ..ab_testing.experiment_manager import ABTestConfig
            ab_config = ABTestConfig(
                experiment_name=f"{config.name}_ab_test",
                control_params=best_params,
                treatment_params={
                    **best_params, 
                    'learning_rate': best_params.get('learning_rate', 0.001) * 1.2  # 20% increase
                },
                sample_size_per_group=2
            )
            
            # Run A/B test
            ab_experiment_id = await self.ab_test_manager.create_ab_test(ab_config)
            logger.info(f"Created A/B test experiment: {ab_experiment_id}")
            
            # Wait for A/B test completion
            await asyncio.sleep(5)  # A/B tests complete quickly in simulation
            
            # Simulate results (in production, this would come from actual testing)
            control_performance = best_params.get('learning_rate', 0.001) * 100  # Mock metric
            treatment_performance = control_performance * 1.15  # Treatment wins by 15%
            
            winner = "treatment" if treatment_performance > control_performance else "control"
            logger.info(f"A/B test completed. Winner: {winner}")
            
            return {
                "ab_experiment_id": ab_experiment_id,
                "control_params": ab_config.control_params,
                "treatment_params": ab_config.treatment_params,
                "control_performance": control_performance,
                "treatment_performance": treatment_performance,
                "winner": winner,
                "improvement": (treatment_performance - control_performance) / control_performance * 100
            }
        
        else:
            logger.info(f"Stage {stage.value} not implemented, skipping")
            return {"status": "skipped"}

    # async def _execute_stage(self, stage: PipelineStage, config: PipelineConfig) -> Dict[str, Any]:
        """Execute individual pipeline stage"""
        
        if stage == PipelineStage.DATA_VALIDATION:
            logger.info(f"Validating {config.dataset} dataset")
            await asyncio.sleep(1)  # Simulate validation
            return {"validation_passed": True}
        else:
            logger.info(f"Stage {stage.value} not implemented, skipping")
            return {"status": "skipped"}