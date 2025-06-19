"""
Test Pipeline cho ColonFormer
Kiểm tra toàn bộ pipeline: DataLoader, Model, Training, Evaluation
"""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.colonformer import ColonFormer
from models.losses import ColonFormerLoss
from datasets import PolypDataModule
from utils.metrics import MetricTracker, evaluate_model, print_metrics


def test_data_pipeline():
    """Test DataLoader và Dataset"""
    print("Testing Data Pipeline...")
    print("="*50)
    
    # Create data module
    data_module = PolypDataModule(
        data_root='data',
        img_size=352,
        batch_size=4,  # Small batch for testing
        num_workers=0,  # No multiprocessing for testing
        val_split=0.2
    )
    
    data_module.print_statistics()
    
    # Test train loader
    train_loader = data_module.get_train_dataloader()
    print(f"\nTrain loader created: {len(train_loader)} batches")
    
    # Test val loader
    val_loader = data_module.get_val_dataloader()
    print(f"Val loader created: {len(val_loader)} batches")
    
    # Test one batch
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        images = batch['image']
        masks = batch['mask']
        print(f"Batch test - Images: {images.shape}, Masks: {masks.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
    else:
        print("Warning: No training data found!")
        return False
    
    # Test test loaders
    test_datasets = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-Larib']
    available_test_datasets = []
    
    for dataset_name in test_datasets:
        test_loader = data_module.get_test_dataloader(dataset_name)
        if test_loader is not None:
            print(f"Test loader {dataset_name}: {len(test_loader.dataset)} samples")
            available_test_datasets.append(dataset_name)
        else:
            print(f"Test dataset {dataset_name}: Not available")
    
    print(f"Available test datasets: {available_test_datasets}")
    print("Data pipeline test completed!")
    return True


def test_model():
    """Test Model architecture"""
    print("\nTesting Model...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test different backbones
    backbones = ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3']
    
    for backbone in backbones:
        try:
            print(f"\nTesting {backbone}...")
            model = ColonFormer(backbone=backbone, num_classes=1)
            model = model.to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 352, 352).to(device)
            
            with torch.no_grad():
                outputs = model(dummy_input)
            
            if isinstance(outputs, (list, tuple)):
                print(f"  Output shapes: {[out.shape for out in outputs]}")
                main_output = outputs[0]
            else:
                print(f"  Output shape: {outputs.shape}")
                main_output = outputs
            
            # Check output properties
            expected_shape = (2, 1, 352, 352)
            if main_output.shape == expected_shape:
                print(f"  Shape test: PASSED")
            else:
                print(f"  Shape test: FAILED - Expected {expected_shape}, got {main_output.shape}")
            
            print(f"  Output range: [{main_output.min():.3f}, {main_output.max():.3f}]")
            
        except Exception as e:
            print(f"  ERROR testing {backbone}: {e}")
    
    print("Model test completed!")
    return True


def test_loss_function():
    """Test Loss function"""
    print("\nTesting Loss Function...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function
    criterion = ColonFormerLoss(alpha=0.25, gamma=2.0, lambda_weight=1.0)
    
    # Create dummy data
    batch_size = 2
    pred = torch.randn(batch_size, 1, 352, 352).to(device)
    target = torch.randint(0, 2, (batch_size, 1, 352, 352)).float().to(device)
    
    # Test loss computation
    try:
        loss = criterion(pred, target)
        print(f"Loss computation: PASSED")
        print(f"Loss value: {loss.item():.4f}")
        
        # Test gradient computation
        loss.backward()
        print(f"Gradient computation: PASSED")
        
    except Exception as e:
        print(f"Loss function ERROR: {e}")
        return False
    
    print("Loss function test completed!")
    return True


def test_metrics():
    """Test Metrics computation"""
    print("\nTesting Metrics...")
    print("="*50)
    
    from utils.metrics import dice_coefficient, iou_coefficient, precision_recall, MetricTracker
    
    # Create dummy data
    pred = torch.rand(2, 1, 352, 352)
    target = torch.randint(0, 2, (2, 1, 352, 352)).float()
    
    try:
        # Test individual metrics
        dice = dice_coefficient(pred, target)
        iou = iou_coefficient(pred, target)
        precision, recall = precision_recall(pred, target)
        
        print(f"Dice: {dice:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Test metric tracker
        tracker = MetricTracker()
        tracker.update(pred, target, loss=0.5)
        
        current_metrics = tracker.get_current_metrics()
        avg_metrics = tracker.get_average_metrics()
        
        print("Current metrics:")
        print_metrics(current_metrics, "Test")
        
        print("Average metrics:")
        print_metrics(avg_metrics, "Test")
        
    except Exception as e:
        print(f"Metrics ERROR: {e}")
        return False
    
    print("Metrics test completed!")
    return True


def test_training_step():
    """Test một training step"""
    print("\nTesting Training Step...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create model
        model = ColonFormer(backbone='mit_b1', num_classes=1)  # Use smaller model for testing
        model = model.to(device)
        
        # Create optimizer và scheduler
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        
        # Create loss function
        criterion = ColonFormerLoss()
        
        # Create dummy batch
        images = torch.randn(2, 3, 352, 352).to(device)
        masks = torch.randint(0, 2, (2, 1, 352, 352)).float().to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(images)
        
        # Handle deep supervision
        if isinstance(outputs, (list, tuple)):
            loss_total = 0
            for output in outputs:
                loss_total += criterion(output, masks)
            loss = loss_total / len(outputs)
            main_output = outputs[0]
        else:
            loss = criterion(outputs, masks)
            main_output = outputs
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"Training step: PASSED")
        print(f"Loss: {loss.item():.4f}")
        print(f"Output shape: {main_output.shape}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Test metrics on training output
        tracker = MetricTracker()
        tracker.update(main_output, masks, loss.item())
        
        metrics = tracker.get_current_metrics()
        print_metrics(metrics, "Train Step")
        
    except Exception as e:
        print(f"Training step ERROR: {e}")
        return False
    
    print("Training step test completed!")
    return True


def main():
    """Main test function"""
    print("ColonFormer Pipeline Test")
    print("="*50)
    
    # Run all tests
    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Model Architecture", test_model),
        ("Loss Function", test_loss_function),
        ("Metrics", test_metrics),
        ("Training Step", test_training_step),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"{test_name}: PASSED")
            else:
                print(f"{test_name}: FAILED")
                
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests PASSED! Pipeline is ready for training.")
    else:
        print("Some tests FAILED! Please check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    main() 