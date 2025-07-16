"""
Test script to verify DMIAD model can be trained
"""

import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace

# Import your model
from models.dmiad import build_dmiad_model
from models.losses.combined_loss import CombinedLoss


def create_test_config():
    """Create test configuration"""
    config = SimpleNamespace()
    
    # Dataset config
    config.DATASET = SimpleNamespace()
    config.DATASET.NAME = 'mvtec'
    config.DATASET.ROOT = '/path/to/data'
    config.DATASET.CLASS_NAME = 'bottle'
    config.DATASET.SETTING = 'single'
    config.DATASET.IMAGE_SIZE = (256, 256)
    config.DATASET.RESIZE = 329
    config.DATASET.CROP_SIZE = 288
    
    # Model config
    config.MODEL = SimpleNamespace()
    config.MODEL.NAME = 'dmiad'
    config.MODEL.BACKBONE = 'wide_resnet'
    config.MODEL.PRETRAINED = False
    
    # Memory config
    config.MODEL.MEMORY = SimpleNamespace()
    config.MODEL.MEMORY.TEMPORAL_DIM = 2000
    config.MODEL.MEMORY.SPATIAL_DIM = 2000
    config.MODEL.MEMORY.USE_SPATIAL = True
    config.MODEL.MEMORY.FUSION_METHOD = 'add'
    config.MODEL.MEMORY.SHRINK_THRES = 0.0025
    config.MODEL.MEMORY.NORMALIZE_MEMORY = False
    config.MODEL.MEMORY.NORMALIZE_QUERY = False
    config.MODEL.MEMORY.USE_SHARED_MLP = False
    
    # Training config
    config.TRAIN = SimpleNamespace()
    config.TRAIN.BATCH_SIZE = 4
    config.TRAIN.EPOCHS = 100
    config.TRAIN.LR = 1e-4
    config.TRAIN.WEIGHT_DECAY = 1e-4
    config.TRAIN.SCHEDULER = 'cosine'
    config.TRAIN.WARMUP_EPOCHS = 10
    
    # Loss config
    config.LOSS = SimpleNamespace()
    config.LOSS.RECONSTRUCTION_WEIGHT = 1.0
    config.LOSS.MEMORY_WEIGHT = 0.01
    config.LOSS.SPARSITY_WEIGHT = 0.0001
    
    # Test config
    config.TEST = SimpleNamespace()
    config.TEST.BATCH_SIZE = 1
    config.TEST.SAVE_IMAGES = False
    config.TEST.CHECKPOINT = None
    
    # Augmentation config
    config.AUG = SimpleNamespace()
    config.AUG.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    config.AUG.NORMALIZE_STD = [0.229, 0.224, 0.225]
    config.AUG.HORIZONTAL_FLIP = 0.5
    config.AUG.ROTATION = 10
    config.AUG.COLOR_JITTER = 0.1
    
    # Output config
    config.OUTPUT_DIR = './results'
    config.EXP_NAME = 'test_dmiad'
    config.SAVE_FREQ = 10
    config.EVAL_FREQ = 5
    
    # Device config
    config.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config.NUM_WORKERS = 4
    config.PIN_MEMORY = True
    
    return config


def test_model_creation():
    """Test model creation"""
    print("=== Testing Model Creation ===")
    
    config = create_test_config()
    
    try:
        # Create model
        model = build_dmiad_model(config)
        print(f"✓ Model created successfully: {model.__class__.__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        return model, config
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        raise


def test_forward_pass(model, config):
    """Test forward pass"""
    print("\n=== Testing Forward Pass ===")
    
    try:
        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, config.DATASET.CROP_SIZE, config.DATASET.CROP_SIZE)
        
        print(f"✓ Input shape: {dummy_input.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Output keys: {list(outputs.keys())}")
        print(f"✓ Reconstructed shape: {outputs['reconstructed'].shape}")
        
        # Test anomaly scoring
        anomaly_scores = model.compute_anomaly_score(dummy_input, outputs['reconstructed'])
        print(f"✓ Anomaly scoring successful")
        print(f"✓ Image scores shape: {anomaly_scores['image_scores'].shape}")
        print(f"✓ Pixel scores shape: {anomaly_scores['pixel_scores'].shape}")
        
        return outputs, anomaly_scores
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_loss_computation(model, config):
    """Test loss computation"""
    print("\n=== Testing Loss Computation ===")
    
    try:
        # Create loss function
        criterion = CombinedLoss(config)
        print(f"✓ Loss function created: {criterion.__class__.__name__}")
        
        # Create dummy data
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, config.DATASET.CROP_SIZE, config.DATASET.CROP_SIZE)
        dummy_labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        model.train()
        outputs = model(dummy_input)
        
        # Compute loss
        loss_dict = criterion(
            dummy_input,
            outputs['reconstructed'],
            outputs['memory_results'],
            dummy_labels
        )
        
        print(f"✓ Loss computation successful")
        print(f"✓ Loss components: {list(loss_dict.keys())}")
        print(f"✓ Total loss: {loss_dict['total_loss'].item():.6f}")
        
        return loss_dict
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        raise


def test_backward_pass(model, config):
    """Test backward pass and optimization"""
    print("\n=== Testing Backward Pass ===")
    
    try:
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.LR)
        criterion = CombinedLoss(config)
        
        # Create dummy data
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, config.DATASET.CROP_SIZE, config.DATASET.CROP_SIZE)
        dummy_labels = torch.randint(0, 2, (batch_size,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(dummy_input)
        loss_dict = criterion(
            dummy_input,
            outputs['reconstructed'],
            outputs['memory_results'],
            dummy_labels
        )
        
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        optimizer.step()
        
        print(f"✓ Backward pass successful")
        print(f"✓ Gradients computed and parameters updated")
        
        # Check gradients
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"✓ Parameters have gradients: {has_gradients}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        raise


def test_memory_functions(model):
    """Test memory-related functions"""
    print("\n=== Testing Memory Functions ===")
    
    try:
        # Get memory items
        memory_items = model.get_memory_items()
        print(f"✓ Memory items retrieved: {list(memory_items.keys())}")
        
        for name, item in memory_items.items():
            print(f"  - {name}: {item.shape}")
        
        return memory_items
        
    except Exception as e:
        print(f"✗ Memory function test failed: {e}")
        raise


def test_mini_training_loop(model, config, num_steps=5):
    """Test a mini training loop"""
    print(f"\n=== Testing Mini Training Loop ({num_steps} steps) ===")
    
    try:
        # Setup training
        optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.LR)
        criterion = CombinedLoss(config)
        
        model.train()
        
        for step in range(num_steps):
            # Create dummy data
            batch_size = 2
            dummy_input = torch.randn(batch_size, 3, config.DATASET.CROP_SIZE, config.DATASET.CROP_SIZE)
            dummy_labels = torch.randint(0, 2, (batch_size,))
            
            # Training step
            optimizer.zero_grad()
            
            outputs = model(dummy_input)
            loss_dict = criterion(
                dummy_input,
                outputs['reconstructed'],
                outputs['memory_results'],
                dummy_labels
            )
            
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            optimizer.step()
            
            print(f"  Step {step+1}/{num_steps}: Loss = {total_loss.item():.6f}")
        
        print(f"✓ Mini training loop completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Mini training loop failed: {e}")
        raise


def main():
    """Main test function"""
    print("DMIAD Model Training Test")
    print("=" * 50)
    
    try:
        # Test 1: Model Creation
        model, config = test_model_creation()
        
        # Test 2: Forward Pass
        outputs, anomaly_scores = test_forward_pass(model, config)
        
        # Test 3: Loss Computation
        loss_dict = test_loss_computation(model, config)
        
        # Test 4: Backward Pass
        test_backward_pass(model, config)
        
        # Test 5: Memory Functions
        memory_items = test_memory_functions(model)
        
        # Test 6: Mini Training Loop
        test_mini_training_loop(model, config)
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Model is ready for training.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("=" * 50)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)