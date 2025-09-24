import sys
import os
import asyncio
import logging
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.master.grpc_coordinator import GRPCCoordinator
from src.parameter_server.parameter_server import ParameterServer
from src.worker.production_ml_worker import ProductionMLWorker
from src.utils.config import ConfigManager
from src.data.distributed_loader import MNISTDataModule
from src.models.production_models import CNN_MNIST, get_model
from src.checkpoint.checkpoint_manager import DistributedCheckpointManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_dataset_loading():
    """Test: Real datasets can be loaded and distributed"""
    print("Testing real dataset loading...")
    
    try:
        data_module = MNISTDataModule(batch_size=16)
        data_module.prepare_data()
        
        # Test distributed data loading
        dataloader = data_module.get_train_dataloader("test_worker_0", 2)
        
        # Get a batch
        data_iter = iter(dataloader)
        batch_data, batch_labels = next(data_iter)
        
        assert batch_data.shape[0] <= 16, "Batch size incorrect"
        assert batch_data.shape[1:] == (1, 28, 28), "MNIST shape incorrect"
        assert batch_labels.shape[0] <= 16, "Label batch size incorrect"
        
        print("✓ Real dataset loading working")
        
    except Exception as e:
        raise AssertionError(f"Dataset loading failed: {e}")

async def test_production_models():
    """Test: Production models can be created and used"""
    print("Testing production model architectures...")
    
    try:
        # Test CNN for MNIST
        model = get_model("cnn", "mnist")
        assert isinstance(model, CNN_MNIST), "CNN model not created correctly"
        
        # Test model forward pass
        dummy_input = torch.randn(1, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (1, 10), "Model output shape incorrect"
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1000, "Model too small"
        
        print("✓ Production model architectures working")
        
    except Exception as e:
        raise AssertionError(f"Production model test failed: {e}")

async def test_checkpoint_system():
    """Test: Checkpointing system works"""
    print("Testing checkpoint system...")
    
    try:
        checkpoint_manager = DistributedCheckpointManager(worker_id="test_worker")
        
        # Create dummy model state
        model = CNN_MNIST()
        model_state = model.state_dict()
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model_state=model_state,
            epoch=1,
            loss=0.5
        )
        
        assert checkpoint_path != "", "Checkpoint not saved"
        assert os.path.exists(checkpoint_path), "Checkpoint file not created"
        
        # Load checkpoint
        loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
        assert loaded_checkpoint is not None, "Checkpoint not loaded"
        assert loaded_checkpoint['epoch'] == 1, "Checkpoint data incorrect"
        
        print("✓ Checkpoint system working")
        
        # Cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            
    except Exception as e:
        raise AssertionError(f"Checkpoint test failed: {e}")

async def test_production_worker_integration():
    """Test: Production worker integrates all components"""
    print("Testing production worker integration...")
    
    config = ConfigManager()
    
    # Start services
    coordinator = GRPCCoordinator(config)
    param_server = ParameterServer(config)
    
    master_task = asyncio.create_task(coordinator.start())
    ps_task = asyncio.create_task(param_server.start())
    await asyncio.sleep(2)
    
    try:
        # Start production worker
        worker = ProductionMLWorker("prod_test_worker", config, dataset="mnist", model_name="cnn")
        
        # Test initialization
        assert worker.dataset == "mnist", "Dataset not set correctly"
        assert worker.model_name == "cnn", "Model name not set correctly"
        assert worker.data_module is not None, "Data module not initialized"
        assert worker.model is not None, "Model not initialized"
        
        # Test connections (don't run full training)
        await worker.connect_to_master()
        await worker.connect_to_parameter_server()
        await worker.register_with_master()
        await worker.register_with_ps()
        
        # Check registrations
        assert "prod_test_worker" in coordinator.workers, "Worker not registered with master"
        
        print("✓ Production worker integration working")
        
        await worker.stop()
        
    except Exception as e:
        raise AssertionError(f"Production worker integration failed: {e}")
    
    finally:
        await param_server.stop()
        await coordinator.stop()
        ps_task.cancel()
        master_task.cancel()

async def test_distributed_data_splitting():
    """Test: Data is properly split between workers"""
    print("Testing distributed data splitting...")
    
    try:
        data_module = MNISTDataModule(batch_size=32)
        data_module.prepare_data()
        
        # Get dataloaders for 2 workers
        dataloader1 = data_module.get_train_dataloader("worker_0", 2)
        dataloader2 = data_module.get_train_dataloader("worker_1", 2)
        
        # Check that they have different data ranges
        sampler1 = dataloader1.sampler
        sampler2 = dataloader2.sampler
        
        assert sampler1.start_idx != sampler2.start_idx, "Workers have same data range"
        assert sampler1.end_idx != sampler2.end_idx, "Workers have same data range"
        
        # Check total coverage
        total_coverage = (sampler1.end_idx - sampler1.start_idx) + (sampler2.end_idx - sampler2.start_idx)
        expected_coverage = len(data_module.train_dataset)
        
        # Should cover most of the dataset (allowing for integer division)
        assert total_coverage >= expected_coverage * 0.9, "Data coverage insufficient"
        
        print("✓ Distributed data splitting working")
        
    except Exception as e:
        raise AssertionError(f"Distributed data splitting failed: {e}")

async def run_phase3_tests():
    """Run all Phase 3 tests"""
    print("Starting Phase 3 Verification Tests...\n")
    
    try:
        await test_real_dataset_loading()
        await test_production_models()
        await test_checkpoint_system()
        await test_distributed_data_splitting()
        await test_production_worker_integration()
        
        print("\n🎉 ALL PHASE 3 TESTS PASSED!")
        print("✅ Real dataset loading functional")
        print("✅ Production model architectures working")
        print("✅ Checkpoint system operational")
        print("✅ Distributed data splitting working")
        print("✅ Production worker integration successful")
        print("\nPhase 3 production ML pipeline is fully functional!")
        
    except Exception as e:
        print(f"\n❌ PHASE 3 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_phase3_tests())