# ColonFormer - Clean Version

ColonFormer implementation kế thừa từ code gốc với clean training và testing pipeline.

## Tổng quan

- **train.py**: Training script chính với full configuration options
- **test.py**: Testing script cho toàn bộ TestDataset với metrics đầy đủ
- **Không có UI**: Chỉ sử dụng command line arguments cho configuration
- **Config logging**: Tất cả config được lưu và log trong quá trình training/testing

## Cài đặt

1. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

2. Chuẩn bị data:

- Training data: `data/TrainDataset/image/` và `data/TrainDataset/mask/`
- Test data: `data/TestDataset/[CVC-300|CVC-ClinicDB|CVC-ColonDB|ETIS-LaribPolypDB|Kvasir]/images/` và `masks/`

## Training

### Basic Usage

```bash
python train.py --data-root data/TrainDataset --epochs 100
```

### Full Configuration Options

#### Model Options

- `--backbone`: Backbone type (default: 'simple')
- `--use-refinement`: Enable refinement modules (default: True)
- `--num-classes`: Number of classes (default: 1)

#### Training Options

- `--epochs`: Number of epochs (default: 100)
- `--batch-size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 1e-4)
- `--img-size`: Image size (default: 352)

#### Loss Options (QUAN TRỌNG - bao gồm crossentropy)

- `--loss-type`: Loss function ['structure', 'focal', 'crossentropy', 'ce'] (default: 'structure')
- `--focal-alpha`: Focal loss alpha (default: 0.25)
- `--focal-gamma`: Focal loss gamma (default: 2.0)
- `--iou-weight`: IoU loss weight (default: 1.0)

#### Optimizer Options

- `--optimizer`: Optimizer type ['adamw', 'adam', 'sgd'] (default: 'adamw')
- `--scheduler`: Scheduler type ['cosine', 'step', 'poly'] (default: 'cosine')
- `--momentum`: SGD momentum (default: 0.9)

#### Data Options

- `--data-root`: Dataset root directory (default: 'data/TrainDataset')
- `--val-split`: Validation split ratio (default: 0.2)
- `--num-workers`: Number of workers (default: 2)

#### Training Options

- `--work-dir`: Work directory (default: 'work_dirs/colonformer')
- `--resume-from`: Resume from checkpoint
- `--seed`: Random seed (default: 42)

### Ví dụ Training Commands

1. **Basic training với structure loss:**

```bash
python train.py --epochs 50 --batch-size 8 --lr 2e-4
```

2. **Training với CrossEntropy loss:**

```bash
python train.py --loss-type crossentropy --epochs 100 --lr 1e-4
```

3. **Training với Focal loss:**

```bash
python train.py --loss-type focal --focal-alpha 0.5 --focal-gamma 1.5
```

4. **Training với SGD optimizer:**

```bash
python train.py --optimizer sgd --lr 1e-3 --momentum 0.95
```

5. **Full custom training:**

```bash
python train.py \
  --epochs 200 \
  --batch-size 16 \
  --lr 5e-4 \
  --loss-type structure \
  --iou-weight 2.0 \
  --optimizer adamw \
  --scheduler cosine \
  --img-size 384 \
  --work-dir work_dirs/custom_experiment
```

## Testing

### Basic Usage

```bash
python test.py --checkpoint work_dirs/colonformer/best_model.pth
```

### Test Options

- `--checkpoint`: Path to model checkpoint (required)
- `--test-data`: Path to test dataset (default: 'data/TestDataset')
- `--output-dir`: Output directory for results (default: 'test_results')
- `--batch-size`: Test batch size (default: 8)
- `--img-size`: Input image size (default: 352)
- `--num-workers`: Number of workers (default: 2)

### Ví dụ Test Commands

1. **Basic testing:**

```bash
python test.py --checkpoint work_dirs/colonformer/best_model.pth
```

2. **Test với custom settings:**

```bash
python test.py \
  --checkpoint work_dirs/custom_experiment/best_model.pth \
  --test-data data/TestDataset \
  --batch-size 16 \
  --img-size 384 \
  --output-dir results/custom_test
```

## Outputs

### Training Outputs

- **Model checkpoints**: `work_dirs/colonformer/best_model.pth`, `checkpoint_epoch_*.pth`
- **Config file**: `work_dirs/colonformer/config.json`
- **TensorBoard logs**: `work_dirs/colonformer/tensorboard/`
- **Console logs**: Real-time training progress với config display

### Testing Outputs

- **Console results**: Detailed metrics per dataset và overall
- **JSON results**: `test_results/test_results_TIMESTAMP.json`

Ví dụ test output:

```
COLONFORMER TEST RESULTS
========================

CONFIGURATION SUMMARY:
  Model: ColonFormer (backbone: simple)
  Refinement: Enabled
  Loss Function: structure
  Image Size: 352x352
  Learning Rate: 0.0001
  Batch Size: 4
  Checkpoint Epoch: 50
  Best Training Dice: 0.8543

OVERALL RESULTS:
  Dice Score:  0.8234
  IoU Score:   0.7456
  Precision:   0.8567
  Recall:      0.7898

PER-DATASET RESULTS:

  CVC-300 (60 samples):
    Dice:      0.8456
    IoU:       0.7612
    Precision: 0.8723
    Recall:    0.8234

  CVC-ClinicDB (62 samples):
    Dice:      0.8123
    IoU:       0.7345
    Precision: 0.8456
    Recall:    0.7789

  [... other datasets ...]
```

## Model Architecture

ColonFormer kế thừa CHÍNH XÁC từ code gốc:

1. **Backbone**: SimpleBackbone (3→64→128→320→512 channels)
2. **Decode Head**: SimpleDecodeHead tương tự SegFormer
3. **Refinement** (nếu enabled):
   - CFPModule từ code gốc
   - RA-RA (Reverse Attention - Residual Attention)
   - AA_kernel (Axial Attention)

## Features

- ✅ **Full parser configuration**: Tất cả settings qua command line
- ✅ **Config logging**: Lưu và display config trong training/testing
- ✅ **Cross Entropy Loss**: Thêm option crossentropy trong optimizer
- ✅ **Complete metrics**: Dice, IoU, Precision, Recall
- ✅ **Multi-dataset testing**: Test trên toàn bộ TestDataset
- ✅ **TensorBoard logging**: Real-time monitoring
- ✅ **Checkpoint management**: Auto save best model
- ✅ **Kế thừa từ code gốc**: 100% logic từ mmseg/models/segmentors/

## File Structure

```
ColonFormer/
├── train.py              # Main training script
├── test.py               # Main testing script
├── README.md             # This file
├── requirements.txt      # Dependencies
├── mmseg/                # Original code modules
│   └── models/segmentors/lib/
│       ├── conv_layer.py
│       ├── axial_atten.py
│       └── context_module.py
├── data/                 # Data directory
│   ├── TrainDataset/
│   └── TestDataset/
└── work_dirs/            # Training outputs
```

## Dependencies Issues

Nếu gặp lỗi mmcv import, có thể:

1. Cài đặt mmcv: `pip install mmcv`
2. Hoặc comment các import mmseg trong code tạm thời để test options

## Notes

- Model parameters: ~53M parameters
- Training time: Depends on hardware và epochs
- Best results với refinement enabled
- CrossEntropy loss option đã được thêm vào optimizer choices
