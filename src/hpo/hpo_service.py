import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from .bayesian_optimizer import BayesianOptimizer
from .experiment_tracker import ExperimentTracker
from ..utils.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HPOConfig:
    experiment_name: str
    search_space: Dict[str, Dict[str, Any]]
    objective_metric: str
    optimization_direction: str = "maximize"  # or "minimize"
    max_trials: int = 20
    max_parallel_trials: int = 4
    early_stopping_patience: int = 3

@dataclass
class TrialConfig:
    trial_id: str
    experiment_id: str
    parameters: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    metrics: Dict[str, float] = None

class HPOService:
    """Hyperparameter Optimization Service"""
    
    def __init__(self, coordinator, config: ConfigManager):
        self.coordinator = coordinator  # Your existing GRPCCoordinator
        self.config = config
        self.experiments: Dict[str, Dict] = {}
        self.active_trials: Dict[str, TrialConfig] = {}
        self.experiment_tracker = ExperimentTracker()
        self.experiment_tasks: Dict[str, asyncio.Task] = {}  # Keep task references!
        
    async def create_experiment(self, hpo_config: HPOConfig) -> str:
        """Create new HPO experiment"""
        experiment_id = str(uuid.uuid4())
        
        # Initialize Bayesian optimizer
        optimizer = BayesianOptimizer(
            search_space=hpo_config.search_space,
            objective=hpo_config.objective_metric,
            direction=hpo_config.optimization_direction
        )
        
        experiment_data = {
            'id': experiment_id,
            'config': hpo_config,
            'optimizer': optimizer,
            'trials': {},
            'best_trial': None,
            'status': 'active',
            'completed_trials': 0,
            'processed_trials': set()
        }
        
        self.experiments[experiment_id] = experiment_data
        logger.info(f"Created HPO experiment: {experiment_id}")
        
        # CRITICAL: Store the task reference to prevent garbage collection
        experiment_task = asyncio.create_task(self._run_experiment(experiment_id))
        self.experiment_tasks[experiment_id] = experiment_task
        
        return experiment_id
    
    # async def _run_experiment(self, experiment_id: str):
        """Run HPO experiment with proper multi-trial handling"""
        try:
            experiment = self.experiments[experiment_id]
            config = experiment['config']
            optimizer = experiment['optimizer']

            logger.info(f"🔬 Starting experiment {experiment_id[:8]} with {config.max_trials} trials "
                        f"(parallel={config.max_parallel_trials})")

            trials_started = 0
            loop_count = 0
            
            # Main experiment loop
            while experiment['status'] == 'active' and loop_count < 500:
                loop_count += 1
                
                # Get current trial statuses directly from experiment
                all_trials = list(experiment['trials'].values())
                completed_trials = [t for t in all_trials if t.status == 'completed']
                failed_trials = [t for t in all_trials if t.status == 'failed']
                running_trials = [t for t in all_trials if t.status == 'running']
                
                completed_count = len(completed_trials)
                failed_count = len(failed_trials)
                running_count = len(running_trials)

                # Log progress every 10 iterations
                if loop_count % 10 == 1:
                    logger.info(f"📊 Loop {loop_count}: {running_count} running, {completed_count} completed, "
                               f"{failed_count} failed, {trials_started} started (target: {config.max_trials})")

                # Process completed trials
                for trial in completed_trials:
                    if trial.trial_id not in experiment['processed_trials']:
                        logger.info(f"🔄 Processing completed trial {trial.trial_id[:8]}")
                        
                        try:
                            if trial.metrics:
                                # Report to optimizer
                                await optimizer.report(trial.parameters, trial.metrics)
                                # Update best trial
                                await self._update_best_trial(experiment_id, trial)
                                logger.info(f"✅ Successfully processed trial {trial.trial_id[:8]}: {trial.metrics}")
                            
                            # Mark as processed
                            experiment['processed_trials'].add(trial.trial_id)
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing trial {trial.trial_id[:8]}: {e}")

                # Start new trials if we have capacity
                can_start_more = (running_count < config.max_parallel_trials and 
                                trials_started < config.max_trials)
                
                if can_start_more:
                    try:
                        # Get new parameters
                        suggested_params = await optimizer.suggest()
                        
                        # Create new trial
                        trial_id = str(uuid.uuid4())
                        trial = TrialConfig(
                            trial_id=trial_id,
                            experiment_id=experiment_id,
                            parameters=suggested_params,
                            status="running"
                        )

                        # Add to experiment
                        experiment['trials'][trial_id] = trial
                        trials_started += 1

                        # Start trial execution
                        trial_task = asyncio.create_task(self._run_trial(trial))
                        logger.info(f"🚀 Started trial {trials_started}/{config.max_trials}: "
                                   f"{trial_id[:8]} with params: {suggested_params}")

                        # Brief pause after starting trial
                        await asyncio.sleep(1)

                    except Exception as e:
                        logger.error(f"❌ Failed to start new trial: {e}")
                        await asyncio.sleep(5)

                # Check completion
                total_finished = completed_count + failed_count
                if total_finished >= config.max_trials:
                    logger.info(f"🎯 Target reached: {completed_count} completed, {failed_count} failed")
                    break
                
                # If no trials are running and we can't start more, we're done
                if running_count == 0 and not can_start_more:
                    logger.info(f"🏁 No more trials to run: {completed_count} completed, {failed_count} failed")
                    break

                # Wait before next loop iteration
                await asyncio.sleep(3)  # Increased wait time for stability

            # Mark as completed
            experiment['status'] = 'completed'
            
            # Final status
            final_trials = list(experiment['trials'].values())
            final_completed = len([t for t in final_trials if t.status == 'completed'])
            final_failed = len([t for t in final_trials if t.status == 'failed'])
            
            logger.info(f"🏁 EXPERIMENT {experiment_id[:8]} COMPLETED: "
                       f"{final_completed} successful, {final_failed} failed trials")

            if experiment['best_trial']:
                best = experiment['best_trial']
                logger.info(f"🏆 BEST RESULT: {best.parameters} -> {best.metrics}")

        except Exception as e:
            logger.error(f"💥 Experiment {experiment_id[:8]} crashed: {e}")
            import traceback
            traceback.print_exc()
            experiment['status'] = 'failed'
        
        finally:
            # Clean up task reference
            self.experiment_tasks.pop(experiment_id, None)

    async def _run_experiment(self, experiment_id: str):
        """Run HPO experiment with proper multi-trial handling"""
        try:
            experiment = self.experiments[experiment_id]
            config = experiment['config']
            optimizer = experiment['optimizer']

            logger.info(f"🔬 Starting experiment {experiment_id[:8]} with {config.max_trials} trials "
                        f"(parallel={config.max_parallel_trials})")

            trials_started = 0
            
            # Keep running until we have enough completed trials
            while trials_started < config.max_trials and experiment['status'] == 'active':
                
                # Get current trial statuses
                all_trials = list(experiment['trials'].values())
                completed_trials = [t for t in all_trials if t.status == 'completed']
                failed_trials = [t for t in all_trials if t.status == 'failed']
                running_trials = [t for t in all_trials if t.status == 'running']
                
                completed_count = len(completed_trials)
                failed_count = len(failed_trials)
                running_count = len(running_trials)

                logger.info(f"📊 Status: {running_count} running, {completed_count} completed, "
                        f"{failed_count} failed, {trials_started} started (target: {config.max_trials})")

                # Process completed trials
                for trial in completed_trials:
                    if trial.trial_id not in experiment['processed_trials']:
                        logger.info(f"🔄 Processing completed trial {trial.trial_id[:8]}")
                        
                        try:
                            if trial.metrics:
                                await optimizer.report(trial.parameters, trial.metrics)
                                await self._update_best_trial(experiment_id, trial)
                                logger.info(f"✅ Successfully processed trial {trial.trial_id[:8]}: {trial.metrics}")
                            
                            experiment['processed_trials'].add(trial.trial_id)
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing trial {trial.trial_id[:8]}: {e}")

                # Start new trials if we can
                can_start_more = (running_count < config.max_parallel_trials and 
                                trials_started < config.max_trials)
                
                if can_start_more:
                    try:
                        # Get new parameters
                        suggested_params = await optimizer.suggest()
                        
                        # Create new trial
                        trial_id = str(uuid.uuid4())
                        trial = TrialConfig(
                            trial_id=trial_id,
                            experiment_id=experiment_id,
                            parameters=suggested_params,
                            status="running"
                        )

                        # Add to experiment
                        experiment['trials'][trial_id] = trial
                        trials_started += 1

                        # Start trial execution
                        asyncio.create_task(self._run_trial(trial))
                        logger.info(f"🚀 Started trial {trials_started}/{config.max_trials}: "
                                f"{trial_id[:8]} with params: {suggested_params}")

                        # Brief pause after starting trial
                        await asyncio.sleep(2)

                    except Exception as e:
                        logger.error(f"❌ Failed to start new trial: {e}")
                        await asyncio.sleep(5)

                # Wait before next loop iteration
                await asyncio.sleep(5)

            # Mark as completed
            experiment['status'] = 'completed'
            
            # Final status
            final_trials = list(experiment['trials'].values())
            final_completed = len([t for t in final_trials if t.status == 'completed'])
            final_failed = len([t for t in final_trials if t.status == 'failed'])
            
            logger.info(f"🏁 EXPERIMENT {experiment_id[:8]} COMPLETED: "
                    f"{final_completed} successful, {final_failed} failed trials")

            if experiment['best_trial']:
                best = experiment['best_trial']
                logger.info(f"🏆 BEST RESULT: {best.parameters} -> {best.metrics}")

        except Exception as e:
            logger.error(f"💥 Experiment {experiment_id[:8]} crashed: {e}")
            import traceback
            traceback.print_exc()
            experiment['status'] = 'failed'
        
        finally:
            # Clean up task reference
            self.experiment_tasks.pop(experiment_id, None)

    async def _run_trial(self, trial: TrialConfig):
        """Run a single trial"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"⏳ Starting trial execution: {trial.trial_id[:8]}")
            
            # Create HPO worker
            worker_id = f"hpo_worker_{trial.trial_id[:8]}"
            worker = self._create_hpo_worker(worker_id, trial.parameters)
            
            # Run training
            logger.info(f"🏃 Training trial {trial.trial_id[:8]}...")
            metrics = await worker.train_with_hpo(trial.trial_id)
            
            # Store results
            trial.metrics = metrics
            trial.status = "completed"
            
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(f"✅ Trial {trial.trial_id[:8]} COMPLETED in {duration:.1f}s: {metrics}")
            
        except Exception as e:
            logger.error(f"❌ Trial {trial.trial_id[:8]} failed: {e}")
            trial.status = "failed"
            trial.metrics = {'accuracy': 0.0, 'loss': 1.0}

    def _create_hpo_worker(self, worker_id: str, hyperparams: Dict):
        """Create HPO worker"""
        from ..worker.hpo_ml_worker import HPOMLWorker
        
        return HPOMLWorker(
            worker_id=worker_id,
            config=self.config,
            hyperparameters=hyperparams,
            dataset="mnist",
            model_name="cnn"
        )
    
    async def _update_best_trial(self, experiment_id: str, trial: TrialConfig):
        """Update best trial"""
        experiment = self.experiments[experiment_id]
        
        if not experiment['best_trial'] or self._is_better_trial(
            trial, experiment['best_trial'], experiment['config'].optimization_direction
        ):
            experiment['best_trial'] = trial
            logger.info(f"🆕 NEW BEST: {trial.parameters} -> {trial.metrics}")
    
    def _is_better_trial(self, trial1: TrialConfig, trial2: TrialConfig, direction: str) -> bool:
        """Compare trials"""
        if not trial1.metrics or not trial2.metrics:
            return False
            
        # Find accuracy metric
        metric_key = 'accuracy'
        for key in trial1.metrics:
            if 'accuracy' in key.lower():
                metric_key = key
                break
        
        val1 = trial1.metrics.get(metric_key, 0)
        val2 = trial2.metrics.get(metric_key, 0)
        
        return val1 > val2 if direction == "maximize" else val1 < val2
    
    async def get_experiment_status(self, experiment_id: str) -> Dict:
        """Get experiment status"""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        config = experiment['config']
        
        trials = list(experiment['trials'].values())
        completed = len([t for t in trials if t.status == 'completed'])
        failed = len([t for t in trials if t.status == 'failed'])
        running = len([t for t in trials if t.status == 'running'])
        
        return {
            "experiment_id": experiment_id,
            "status": experiment['status'],
            "completed_trials": completed,
            "failed_trials": failed,
            "running_trials": running,
            "total_trials": config.max_trials,
            "best_trial": asdict(experiment['best_trial']) if experiment['best_trial'] else None
        }
    
    async def stop_experiment(self, experiment_id: str):
        """Stop experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['status'] = 'stopped'
            # Cancel the task if it exists
            if experiment_id in self.experiment_tasks:
                self.experiment_tasks[experiment_id].cancel()
            logger.info(f"Stopped experiment: {experiment_id}")

    async def stop_all_experiments(self):
        """Stop all experiments"""
        for exp_id in list(self.experiments.keys()):
            await self.stop_experiment(exp_id)