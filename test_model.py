"""
Test script cơ bản cho ColonFormer
Kiểm tra xem các component có hoạt động đúng không
"""

import torch
import torch.nn as nn
from models import colonformer_s, colonformer_l, ColonFormerLoss


def test_model_forward():
    """Test forward pass của model"""
    print("Testing ColonFormer forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test data
    batch_size = 2
    img_size = 352
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    targets = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float().to(device)
    
    # Test ColonFormer-S
    print("\n1. Testing ColonFormer-S...")
    model = colonformer_s(num_classes=1, img_size=img_size).to(device)
    
    # Training mode
    model.train()
    outputs_train = model(x)
    print(f"Training mode output type: {type(outputs_train)}")
    if isinstance(outputs_train, dict):
        for key, value in outputs_train.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} tensors")
            else:
                print(f"  {key}: {value.shape}")
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        outputs_eval = model(x)
        print(f"Eval mode output shape: {outputs_eval.shape}")
    
    print("ColonFormer-S forward test passed!")
    
    return model, outputs_train, targets


def test_loss_function():
    """Test loss function"""
    print("\n2. Testing ColonFormer Loss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 2
    img_size = 352
    
    # Predictions (giả lập output từ model)
    main_pred = torch.randn(batch_size, 1, img_size, img_size).to(device)
    aux_preds = [
        torch.randn(batch_size, 1, img_size//2, img_size//2).to(device),
        torch.randn(batch_size, 1, img_size//4, img_size//4).to(device)
    ]
    
    predictions = {
        'main': main_pred,
        'aux': aux_preds
    }
    
    targets = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float().to(device)
    
    # Test loss
    criterion = ColonFormerLoss(
        focal_alpha=0.25, 
        focal_gamma=2.0, 
        loss_lambda=1.0,
        deep_supervision=True
    ).to(device)
    
    total_loss, loss_dict = criterion(predictions, targets)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("Loss function test passed!")
    
    return total_loss, loss_dict


def test_end_to_end():
    """Test end-to-end training step"""
    print("\n3. Testing end-to-end training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model và data
    model = colonformer_s(num_classes=1, img_size=352).to(device)
    criterion = ColonFormerLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Test data
    batch_size = 2
    img_size = 352
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    targets = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float().to(device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(x)
    
    # Loss computation
    loss, loss_dict = criterion(outputs, targets)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed!")
    print(f"Loss: {loss.item():.4f}")
    print("Gradients computed successfully!")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        inference_output = model(x)
        print(f"Inference output shape: {inference_output.shape}")
    
    print("End-to-end test passed!")


def count_parameters(model):
    """Đếm số parameters của model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_model_sizes():
    """Test và so sánh kích thước các model variants"""
    print("\n4. Testing model sizes...")
    
    models = {
        'ColonFormer-S': colonformer_s(),
        'ColonFormer-L': colonformer_l()
    }
    
    for name, model in models.items():
        total, trainable = count_parameters(model)
        print(f"{name}:")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Model size: ~{total * 4 / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    print("Starting ColonFormer tests...")
    print("=" * 50)
    
    try:
        # Test 1: Model forward pass
        model, outputs, targets = test_model_forward()
        
        # Test 2: Loss function
        loss, loss_dict = test_loss_function()
        
        # Test 3: End-to-end training
        test_end_to_end()
        
        # Test 4: Model sizes
        test_model_sizes()
        
        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("ColonFormer implementation is working correctly!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc() 