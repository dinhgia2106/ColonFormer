# Training Monitor - Real-time Progress Tracking

Hệ thống monitoring cho quá trình training ColonFormer với progress bars và real-time visualization.

## Tính năng chính

### 1. Progress Bars với tqdm

- **Real-time progress** cho từng epoch và batch
- **ETA estimation** (thời gian còn lại)
- **Live metrics**: Loss, Dice, IoU, Learning Rate
- **Memory monitoring**: CPU và GPU usage
- **Batch timing**: Thời gian xử lý từng batch

### 2. Training Logger

- **Automatic logging** metrics vào JSON files
- **Real-time plot generation** sau mỗi epoch
- **Training history visualization** với 6 panels
- **Comprehensive summary** khi kết thúc training

### 3. Metrics Tracking

- Train/Validation Loss
- Dice Score và IoU
- Precision và Recall
- Learning Rate schedule
- Time per epoch

## Cách sử dụng

### Training thông thường

```bash
python train.py --epochs 20 --backbone mit_b3 --batch_size 8
```

Bạn sẽ thấy:

```
Epoch 1/20
--------------------------------------------------
Training Epoch 1: 100%|███████████| 50/50 [00:45<00:00, 1.11batch/s, Loss=0.3245, Dice=0.7834, IoU=0.6721, LR=1.0e-04, Time/batch=0.89s, CPU=45.2%]

Validating...
Validation: 100%|███████████| 10/10 [00:05<00:00, 1.89batch/s, Loss=0.2891, Dice=0.8123, IoU=0.7011]

Epoch 1 Summary:
  Time: 52.3s | ETA: 16.5m | LR: 1.0e-04
  Train - Loss: 0.3245, Dice: 0.7834
  Val   - Loss: 0.2891, Dice: 0.8123
  Best Dice: 0.8123 [NEW]
==================================================
```

### Demo và test

```bash
# Xem tính năng
python demo_training_monitor.py

# Test logger với data giả
python demo_training_monitor.py --mode logger

# Demo training ngắn (2 epochs)
python demo_training_monitor.py --mode training
```

## Output files

Sau khi training, bạn sẽ có:

```
checkpoints/
├── logs/
│   ├── experiment_train.json    # Training metrics
│   └── experiment_val.json      # Validation metrics
├── plots/
│   └── experiment_progress.png  # Training visualization
├── best_model.pth              # Best checkpoint
└── checkpoint_epoch_X.pth      # Periodic checkpoints
```

## Visualization

Training plots bao gồm 6 panels:

1. **Loss** (Train vs Val)
2. **Dice Score** (Train vs Val)
3. **IoU Score** (Train vs Val)
4. **Learning Rate** schedule
5. **Precision & Recall** (Train vs Val)
6. **Time per Epoch**

Plots được update real-time sau mỗi epoch và saved dạng high-resolution PNG.

## Dependencies

Đã có sẵn trong `requirements.txt`:

- `tqdm>=4.62.0` - Progress bars
- `matplotlib>=3.5.0` - Plotting
- `psutil>=5.8.0` - Memory monitoring

## Tips

1. **Quan sát ETA** để estimate thời gian training
2. **Monitor memory usage** để adjust batch size
3. **Check plots** để detect overfitting sớm
4. **Use logs** để compare different experiments
5. **Save checkpoints** thường xuyên cho safety
