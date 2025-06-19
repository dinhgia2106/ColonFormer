# ColonFormer: A Novel Transformer-based Model for Colon Polyp Segmentation

This repository contains the official PyTorch implementation of **ColonFormer**, a transformer-based deep learning model specifically designed for accurate colon polyp segmentation in colonoscopy images.

**Paper**: [ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9845389)

## Abstract

ColonFormer introduces a novel architecture that combines the strengths of Vision Transformers (ViT) with specialized components for medical image segmentation. The model features:

- **MiT (Mix Transformer) Encoder**: Hierarchical transformer backbone for multi-scale feature extraction
- **UPer Decoder with PPM**: Pyramid Pooling Module for global context aggregation
- **Channel-wise Feature Pyramid (CFP)**: Multi-scale dilated convolutions for feature enhancement
- **Residual Axial Attention (RA-RA)**: Efficient attention mechanism for boundary refinement
- **Weighted Loss Function**: Distance-based weighting for improved boundary accuracy

## Architecture Overview

```
ColonFormer Architecture:
Input Image (3x352x352)
    ↓
MiT Encoder (B0/B1/B3/B5 variants)
    ↓
Multi-scale Features [F1, F2, F3, F4]
    ↓
UPer Decoder + PPM → Coarse Prediction
    ↓
Refinement Module (CFP + RA-RA) → Final Prediction
    ↓
Output Mask (1x352x352)
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA-compatible GPU (recommended)

### Setup Environment

```bash
# Clone repository
Clone this repo
cd ColonFormer

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
timm>=0.6.0
tqdm>=4.62.0
matplotlib>=3.5.0
Pillow>=8.3.0
albumentations>=1.1.0
psutil>=5.8.0
pandas>=1.3.0
scipy>=1.7.0
```

## Dataset Preparation

Download the following polyp segmentation datasets:

1. Colon dataset: [Download](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)

### Directory Structure

Organize datasets as follows:

```
data/
├── TrainDataset/
│   ├── images/
│   └── masks/
└── TestDataset/
    ├── Kvasir/
    │   ├── images/
    │   └── masks/
    ├── CVC-ClinicDB/
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/
    ├── CVC-300/
    └── ETIS-LaribPolypDB/
```

## Training

### Model Variants

The implementation provides three model variants with different backbone sizes:

| Model          | Backbone | Parameters | Performance                        |
| -------------- | -------- | ---------- | ---------------------------------- |
| ColonFormer-S  | MiT-B1   | ~13M       | Fast training, good performance    |
| ColonFormer-L  | MiT-B3   | ~45M       | Best balance of speed and accuracy |
| ColonFormer-XL | MiT-B5   | ~82M       | Highest accuracy, slower training  |

### Training Commands

#### ColonFormer-S (Small) - Quick Training

```bash
python train.py --backbone mit_b1 --epochs 20 --batch_size 8 --lr 1e-4 --data_root data/TrainDataset --save_dir checkpoints --img_size 352 --val_split 0.2
```

#### ColonFormer-L (Large) - Paper Baseline

```bash
python train.py --backbone mit_b3 --epochs 20 --batch_size 8 --lr 1e-4 --data_root data/TrainDataset --save_dir checkpoints --img_size 352 --val_split 0.2 --deep_supervision --use_refinement
```

#### ColonFormer-XL (Extra Large) - Maximum Performance

```bash
python train.py --backbone mit_b5 --epochs 30 --batch_size 4 --lr 5e-5 --data_root data/TrainDataset --save_dir checkpoints --img_size 352 --val_split 0.2 --deep_supervision --use_refinement --accumulate_grad_batches 2
```

### Training Parameters

| Parameter       | Description               | Default | Paper Setting |
| --------------- | ------------------------- | ------- | ------------- |
| `--backbone`    | MiT backbone variant      | mit_b3  | mit_b3        |
| `--epochs`      | Number of training epochs | 20      | 20            |
| `--batch_size`  | Training batch size       | 8       | 8             |
| `--lr`          | Learning rate             | 1e-4    | 1e-4          |
| `--img_size`    | Input image size          | 352     | 352           |
| `--optimizer`   | Optimizer type            | adam    | adam          |
| `--scheduler`   | LR scheduler              | cosine  | cosine        |
| `--loss_lambda` | Loss combination weight   | 1.0     | 1.0           |
| `--focal_alpha` | Focal loss alpha          | 0.25    | 0.25          |
| `--focal_gamma` | Focal loss gamma          | 2.0     | 2.0           |

### Advanced Training Options

#### Ablation Study (Paper Table 5)

```bash
# Baseline: MiT-B3 + UPer only
python train.py --backbone mit_b3 --epochs 20 --no_refinement

# + CFP: Add Channel-wise Feature Pyramid
python train.py --backbone mit_b3 --epochs 20 --use_cfp --no_rara

# + RA-RA: Full ColonFormer (CFP + RA-RA)
python train.py --backbone mit_b3 --epochs 20 --use_refinement
```

#### Cross-Dataset Training (Paper Table 3)

```bash
# Train on Kvasir, test on others
python train.py --backbone mit_b3 --data_root data/Kvasir --epochs 25

# Train on CVC-ClinicDB, test on others
python train.py --backbone mit_b3 --data_root data/CVC-ClinicDB --epochs 25
```

## Testing and Evaluation

### Automatic Testing System

The repository includes an intelligent testing system that automatically detects untested experiments and evaluates them on multiple datasets.

#### List All Experiments

```bash
python test_experiments.py --list_experiments
```

#### List Untested Experiments

```bash
python test_experiments.py --list_untested
```

#### Auto-Test All Untested Experiments

```bash
python test_experiments.py --test_data_dir data/TestDataset --datasets Kvasir CVC-ClinicDB CVC-ColonDB CVC-300 ETIS-LaribPolypDB
```

#### Test Specific Experiment

```bash
python test_experiments.py --experiment_id abc12345 --save_predictions
```

#### Generate Paper-Style Results Table

```bash
python test_experiments.py --create_paper_table
```

### Manual Evaluation

#### Single Dataset Evaluation

```bash
python evaluate.py --model_path checkpoints/colonformer_mit_b3_abc12345/best_model.pth --test_data data/TestDataset/Kvasir --backbone mit_b3
```

#### Cross-Dataset Evaluation

```bash
python evaluate.py --model_path checkpoints/colonformer_mit_b3_abc12345/best_model.pth --test_data data/TestDataset --backbone mit_b3 --cross_dataset
```

## Experiment Management

### Unique Experiment Tracking

Each training run generates a unique 8-character experiment ID and comprehensive logging:

```
checkpoints/
├── colonformer_mit_b3_abc12345/
│   ├── experiment_config.json    # Complete configuration
│   ├── results.json              # Training metrics
│   ├── best_model.pth           # Best checkpoint
│   ├── logs/                    # Training history
│   └── plots/                   # Training curves
├── test_results_summary.json    # Consolidated test results
├── test_results_summary.csv     # CSV export for analysis
└── paper_results.csv           # Paper-format results table
```

### Configuration Logging

All experiments automatically log:

- **Model Configuration**: Architecture, backbone, parameters
- **Training Configuration**: Optimizer, scheduler, loss function
- **Data Configuration**: Dataset paths, augmentations, splits
- **System Information**: Hardware, software versions
- **Results**: Training curves, best metrics, test results

## Paper Results Reproduction

### Expected Results (mDice scores from paper)

| Method        | Kvasir | CVC-ClinicDB | CVC-ColonDB | CVC-300 | ETIS-Larib |
| ------------- | ------ | ------------ | ----------- | ------- | ---------- |
| ColonFormer-S | 0.918  | 0.794        | 0.709       | 0.707   | 0.628      |
| ColonFormer-L | 0.927  | 0.822        | 0.731       | 0.734   | 0.720      |

### Reproduction Commands

#### Full Paper Reproduction Pipeline

```bash
# 1. Train ColonFormer-S
python train.py --backbone mit_b1 --epochs 20 --batch_size 8

# 2. Train ColonFormer-L
python train.py --backbone mit_b3 --epochs 20 --batch_size 8

# 3. Auto-test all experiments
python test_experiments.py

# 4. Generate paper table
python test_experiments.py --create_paper_table

# 5. View results
python test_experiments.py --summary
```

#### 5-Fold Cross-Validation (Paper Table 2)

```bash
# Will be added in future version
python cross_validation.py --dataset Kvasir --folds 5
```

## Model Components

### MiT Encoder

- Hierarchical Vision Transformer backbone
- Variants: MiT-B0, B1, B2, B3, B4, B5
- Efficient self-attention with spatial reduction
- Multi-scale feature extraction

### UPer Decoder with PPM

- Feature Pyramid Network structure
- Pyramid Pooling Module for global context
- Deep supervision support
- SE attention mechanisms

### Refinement Module

- **Channel-wise Feature Pyramid (CFP)**: Multi-scale dilated convolutions (rates: 1, 3, 5, 9)
- **Residual Axial Attention (RA-RA)**: Height and width axial attention
- Feature enhancement and boundary refinement

### Loss Function

- **Weighted Focal Loss**: Handles class imbalance (alpha=0.25, gamma=2.0)
- **Weighted IoU Loss**: Distance-based boundary weighting
- **Deep Supervision**: Multi-level loss computation

## Ablation Studies

### Component Analysis (Paper Table 5)

Run ablation studies to analyze individual component contributions:

```bash
# Baseline: Encoder + Decoder only
python ablation_study.py --components encoder decoder

# + CFP: Add Channel-wise Feature Pyramid
python ablation_study.py --components encoder decoder cfp

# + RA-RA: Full model
python ablation_study.py --components encoder decoder cfp rara
```

### Backbone Analysis (Paper Table 6)

Compare different MiT backbone variants:

```bash
# Test all backbone variants
for backbone in mit_b0 mit_b1 mit_b2 mit_b3 mit_b4 mit_b5; do
    python train.py --backbone $backbone --epochs 20
done
```

## File Structure

```
ColonFormer/
├── models/                    # Model implementations
│   ├── backbones/            # MiT encoder variants
│   ├── components/           # CFP, RA-RA, PPM, UPer modules
│   ├── losses/              # Loss functions
│   └── colonformer.py       # Main model class
├── datasets/                 # Data loading utilities
│   ├── polyp_dataset.py     # Dataset class
│   └── __init__.py
├── utils/                   # Utilities and metrics
│   ├── metrics.py           # Evaluation metrics
│   ├── logger.py            # Training visualization
│   ├── experiment_tracker.py # Experiment management
│   └── test_manager.py      # Test results management
├── train.py                 # Training script
├── test_experiments.py      # Automated testing system
├── evaluate.py              # Model evaluation
├── ablation_study.py        # Ablation study framework
├── test_model.py           # Basic model testing
├── test_pipeline.py        # Pipeline testing
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Performance Monitoring

### Real-time Training Monitoring

- Progress bars with ETA estimation
- Live metrics display (Loss, Dice, IoU)
- Memory usage monitoring
- Training curve visualization
- Automatic plot generation

### Comprehensive Logging

- JSON export of training history
- CSV results for analysis
- Model configuration preservation
- System information tracking

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{colonformer2022,
  title={ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation},
  author={[Author names from paper]},
  journal={IEEE Access},
  year={2022},
  publisher={IEEE},
  doi={[DOI from paper]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original paper authors for the ColonFormer architecture
- SegFormer authors for the MiT backbone implementation
- UPerNet authors for the decoder architecture
- Medical imaging community for benchmark datasets

## Contact

For questions and issues, please open an issue on GitHub or contact [your-email@domain.com].

## Updates

- **v1.0**: Initial implementation with full paper reproduction
- **v1.1**: Added automated testing system and experiment management
- **v1.2**: Enhanced monitoring and visualization capabilities
