#!/bin/bash

# ColonFormer Paper Results Reproduction Script
# This script trains all model variants with exact paper parameters

echo "=============================================="
echo "ColonFormer Paper Results Reproduction"
echo "=============================================="

# Check if data directory exists
if [ ! -d "data/TrainDataset" ]; then
    echo "Error: Training data not found at data/TrainDataset"
    echo "Please download and organize datasets according to README.md"
    exit 1
fi

# Check if test data exists
if [ ! -d "data/TestDataset" ]; then
    echo "Warning: Test data not found at data/TestDataset"
    echo "Testing will be skipped"
fi

echo "Starting training with paper parameters..."

# Create checkpoints directory
mkdir -p checkpoints

# ColonFormer-S (Small) - MiT-B1 Backbone
echo ""
echo "1. Training ColonFormer-S (MiT-B1)..."
echo "Expected performance: Kvasir ~0.918, CVC-ClinicDB ~0.794"
python train.py \
    --backbone mit_b1 \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --data_root data/TrainDataset \
    --save_dir checkpoints \
    --img_size 352 \
    --val_split 0.2 \
    --deep_supervision \
    --use_refinement \
    --optimizer adam \
    --scheduler cosine \
    --loss_lambda 1.0 \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --weight_decay 1e-4 \
    --log_interval 10

echo "ColonFormer-S training completed!"

# ColonFormer-L (Large) - MiT-B3 Backbone  
echo ""
echo "2. Training ColonFormer-L (MiT-B3)..."
echo "Expected performance: Kvasir ~0.927, CVC-ClinicDB ~0.822"
python train.py \
    --backbone mit_b3 \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --data_root data/TrainDataset \
    --save_dir checkpoints \
    --img_size 352 \
    --val_split 0.2 \
    --deep_supervision \
    --use_refinement \
    --optimizer adam \
    --scheduler cosine \
    --loss_lambda 1.0 \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --weight_decay 1e-4 \
    --log_interval 10

echo "ColonFormer-L training completed!"

# ColonFormer-XL (Extra Large) - MiT-B5 Backbone
echo ""
echo "3. Training ColonFormer-XL (MiT-B5)..."
echo "Expected performance: Highest accuracy but slower training"
python train.py \
    --backbone mit_b5 \
    --epochs 30 \
    --batch_size 4 \
    --lr 5e-5 \
    --data_root data/TrainDataset \
    --save_dir checkpoints \
    --img_size 352 \
    --val_split 0.2 \
    --deep_supervision \
    --use_refinement \
    --optimizer adam \
    --scheduler cosine \
    --loss_lambda 1.0 \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --weight_decay 1e-4 \
    --accumulate_grad_batches 2 \
    --log_interval 10

echo "ColonFormer-XL training completed!"

# Run ablation study for paper Table 5
echo ""
echo "4. Running Ablation Study (Paper Table 5)..."

# Baseline: MiT-B3 + UPer only
echo "Training Baseline (MiT-B3 + UPer)..."
python train.py \
    --backbone mit_b3 \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --data_root data/TrainDataset \
    --save_dir checkpoints \
    --img_size 352 \
    --val_split 0.2 \
    --no_refinement \
    --deep_supervision \
    --experiment_name "baseline_mit_b3_uper"

# MiT-B3 + UPer + CFP
echo "Training MiT-B3 + UPer + CFP..."
python train.py \
    --backbone mit_b3 \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --data_root data/TrainDataset \
    --save_dir checkpoints \
    --img_size 352 \
    --val_split 0.2 \
    --use_cfp \
    --no_rara \
    --deep_supervision \
    --experiment_name "mit_b3_uper_cfp"

echo "Ablation study completed!"

# Auto-test all trained models if test data available
if [ -d "data/TestDataset" ]; then
    echo ""
    echo "5. Auto-testing all trained models..."
    python test_experiments.py \
        --test_data_dir data/TestDataset \
        --datasets Kvasir CVC-ClinicDB CVC-ColonDB CVC-300 ETIS-LaribPolypDB
    
    echo ""
    echo "6. Generating paper-style results table..."
    python test_experiments.py --create_paper_table
    
    echo ""
    echo "7. Results summary..."
    python test_experiments.py --summary
    
    echo ""
    echo "Results saved to:"
    echo "- checkpoints/test_results_summary.csv"
    echo "- checkpoints/paper_results.csv"
else
    echo ""
    echo "Test data not found. Skipping evaluation."
    echo "To test models later, run:"
    echo "python test_experiments.py --test_data_dir data/TestDataset"
fi

echo ""
echo "=============================================="
echo "Paper Results Reproduction Completed!"
echo "=============================================="
echo ""
echo "Trained models:"
echo "- ColonFormer-S (MiT-B1): Fast training, good performance"
echo "- ColonFormer-L (MiT-B3): Best balance, paper baseline"  
echo "- ColonFormer-XL (MiT-B5): Highest accuracy"
echo "- Ablation models: Baseline, +CFP variants"
echo ""
echo "Check checkpoints/ directory for all experiment results"
echo "Use 'python test_experiments.py --summary' to view results" 