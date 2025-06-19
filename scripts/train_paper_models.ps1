# ColonFormer Paper Results Reproduction Script (PowerShell)
# This script trains all model variants with exact paper parameters

Write-Host "=============================================="
Write-Host "ColonFormer Paper Results Reproduction"
Write-Host "=============================================="

# Check if data directory exists
if (-not (Test-Path "data\TrainDataset")) {
    Write-Host "Error: Training data not found at data\TrainDataset" -ForegroundColor Red
    Write-Host "Please download and organize datasets according to README.md"
    exit 1
}

# Check if test data exists
if (-not (Test-Path "data\TestDataset")) {
    Write-Host "Warning: Test data not found at data\TestDataset" -ForegroundColor Yellow
    Write-Host "Testing will be skipped"
}

Write-Host "Starting training with paper parameters..."

# Create checkpoints directory
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

# ColonFormer-S (Small) - MiT-B1 Backbone
Write-Host ""
Write-Host "1. Training ColonFormer-S (MiT-B1)..." -ForegroundColor Green
Write-Host "Expected performance: Kvasir ~0.918, CVC-ClinicDB ~0.794"
python train.py `
    --backbone mit_b1 `
    --epochs 20 `
    --batch_size 8 `
    --lr 1e-4 `
    --data_root data/TrainDataset `
    --save_dir checkpoints `
    --img_size 352 `
    --val_split 0.2 `
    --deep_supervision `
    --use_refinement `
    --optimizer adam `
    --scheduler cosine `
    --loss_lambda 1.0 `
    --focal_alpha 0.25 `
    --focal_gamma 2.0 `
    --weight_decay 1e-4 `
    --log_interval 10

Write-Host "ColonFormer-S training completed!" -ForegroundColor Green

# ColonFormer-L (Large) - MiT-B3 Backbone  
Write-Host ""
Write-Host "2. Training ColonFormer-L (MiT-B3)..." -ForegroundColor Green
Write-Host "Expected performance: Kvasir ~0.927, CVC-ClinicDB ~0.822"
python train.py `
    --backbone mit_b3 `
    --epochs 20 `
    --batch_size 8 `
    --lr 1e-4 `
    --data_root data/TrainDataset `
    --save_dir checkpoints `
    --img_size 352 `
    --val_split 0.2 `
    --deep_supervision `
    --use_refinement `
    --optimizer adam `
    --scheduler cosine `
    --loss_lambda 1.0 `
    --focal_alpha 0.25 `
    --focal_gamma 2.0 `
    --weight_decay 1e-4 `
    --log_interval 10

Write-Host "ColonFormer-L training completed!" -ForegroundColor Green

# ColonFormer-XL (Extra Large) - MiT-B5 Backbone
Write-Host ""
Write-Host "3. Training ColonFormer-XL (MiT-B5)..." -ForegroundColor Green
Write-Host "Expected performance: Highest accuracy but slower training"
python train.py `
    --backbone mit_b5 `
    --epochs 30 `
    --batch_size 4 `
    --lr 5e-5 `
    --data_root data/TrainDataset `
    --save_dir checkpoints `
    --img_size 352 `
    --val_split 0.2 `
    --deep_supervision `
    --use_refinement `
    --optimizer adam `
    --scheduler cosine `
    --loss_lambda 1.0 `
    --focal_alpha 0.25 `
    --focal_gamma 2.0 `
    --weight_decay 1e-4 `
    --accumulate_grad_batches 2 `
    --log_interval 10

Write-Host "ColonFormer-XL training completed!" -ForegroundColor Green

# Run ablation study for paper Table 5
Write-Host ""
Write-Host "4. Running Ablation Study (Paper Table 5)..." -ForegroundColor Cyan

# Baseline: MiT-B3 + UPer only
Write-Host "Training Baseline (MiT-B3 + UPer)..."
python train.py `
    --backbone mit_b3 `
    --epochs 20 `
    --batch_size 8 `
    --lr 1e-4 `
    --data_root data/TrainDataset `
    --save_dir checkpoints `
    --img_size 352 `
    --val_split 0.2 `
    --no_refinement `
    --deep_supervision `
    --experiment_name "baseline_mit_b3_uper"

# MiT-B3 + UPer + CFP
Write-Host "Training MiT-B3 + UPer + CFP..."
python train.py `
    --backbone mit_b3 `
    --epochs 20 `
    --batch_size 8 `
    --lr 1e-4 `
    --data_root data/TrainDataset `
    --save_dir checkpoints `
    --img_size 352 `
    --val_split 0.2 `
    --use_cfp `
    --no_rara `
    --deep_supervision `
    --experiment_name "mit_b3_uper_cfp"

Write-Host "Ablation study completed!" -ForegroundColor Cyan

# Auto-test all trained models if test data available
if (Test-Path "data\TestDataset") {
    Write-Host ""
    Write-Host "5. Auto-testing all trained models..." -ForegroundColor Yellow
    python test_experiments.py `
        --test_data_dir data/TestDataset `
        --datasets Kvasir CVC-ClinicDB CVC-ColonDB CVC-300 ETIS-LaribPolypDB
    
    Write-Host ""
    Write-Host "6. Generating paper-style results table..." -ForegroundColor Yellow
    python test_experiments.py --create_paper_table
    
    Write-Host ""
    Write-Host "7. Results summary..." -ForegroundColor Yellow
    python test_experiments.py --summary
    
    Write-Host ""
    Write-Host "Results saved to:"
    Write-Host "- checkpoints/test_results_summary.csv"
    Write-Host "- checkpoints/paper_results.csv"
} else {
    Write-Host ""
    Write-Host "Test data not found. Skipping evaluation."
    Write-Host "To test models later, run:"
    Write-Host "python test_experiments.py --test_data_dir data/TestDataset"
}

Write-Host ""
Write-Host "=============================================="
Write-Host "Paper Results Reproduction Completed!" -ForegroundColor Green
Write-Host "=============================================="
Write-Host ""
Write-Host "Trained models:"
Write-Host "- ColonFormer-S (MiT-B1): Fast training, good performance"
Write-Host "- ColonFormer-L (MiT-B3): Best balance, paper baseline"  
Write-Host "- ColonFormer-XL (MiT-B5): Highest accuracy"
Write-Host "- Ablation models: Baseline, +CFP variants"
Write-Host ""
Write-Host "Check checkpoints/ directory for all experiment results"
Write-Host "Use 'python test_experiments.py --summary' to view results" 