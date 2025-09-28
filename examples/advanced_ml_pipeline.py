import sys
import os
import asyncio
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.master.grpc_coordinator import GRPCCoordinator
from src.parameter_server.parameter_server import ParameterServer
from src.hpo.hpo_service import HPOService
from src.ab_testing.experiment_manager import ABTestManager
from src.pipelines.pipeline_manager import PipelineManager, PipelineConfig, PipelineStage
from src.utils.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Complete Phase 4 example with HPO, A/B testing, and pipelines"""
    config = ConfigManager()
    
    logger.info("🚀 Starting Complete Phase 4 ML Platform")
    
    # Start infrastructure
    coordinator = GRPCCoordinator(config)
    param_server = ParameterServer(config)
    
    master_task = asyncio.create_task(coordinator.start())
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(3)
    
    try:
        # Initialize Phase 4 services
        hpo_service = HPOService(coordinator, config)
        ab_test_manager = ABTestManager(hpo_service)
        pipeline_manager = PipelineManager(coordinator, hpo_service, ab_test_manager)
        
        # Create end-to-end pipeline
        pipeline_config = PipelineConfig(
            name="mnist_complete_pipeline",
            stages=[
                PipelineStage.DATA_VALIDATION,
                PipelineStage.HPO,
                PipelineStage.AB_TESTING,
                PipelineStage.EVALUATION
            ],
            dataset="mnist",
            model="cnn"
        )
        
        pipeline_id = await pipeline_manager.create_pipeline(pipeline_config)
        logger.info(f"📊 Created complete ML pipeline: {pipeline_id}")
        
        # Monitor pipeline progress
        for i in range(20):  # Monitor for 10 minutes
            await asyncio.sleep(30)
            pipeline = pipeline_manager.pipelines[pipeline_id]
            current_stage = pipeline['current_stage']
            status = pipeline['status']
            
            logger.info(f"Pipeline Progress: Stage {current_stage}, Status: {status}")
            
            if status in ['completed', 'failed']:
                logger.info(f"Pipeline finished with status: {status}")
                logger.info(f"Results: {pipeline['results']}")
                break
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        logger.info("🛑 Shutting down Phase 4 platform...")
        await param_server.stop()
        await coordinator.stop()
        ps_task.cancel()
        master_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())