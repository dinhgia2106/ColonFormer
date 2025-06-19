"""
Test Script cho ColonFormer Experiments
Auto-detect experiments chưa test và chạy test trên các datasets
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2

from models.colonformer import ColonFormer
from datasets import PolypDataset
from utils.metrics import dice_coefficient, iou_coefficient, precision_recall
from utils.experiment_tracker import load_experiment, list_experiments
from utils.test_manager import TestResultsManager, create_paper_results_table


class ModelTester:
    """
    Tester để evaluate trained models trên test datasets
    """
    
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
    
    def load_model_from_experiment(self, experiment_dir):
        """Load model từ experiment directory"""
        # Load experiment config
        exp_data = load_experiment(experiment_dir)
        model_config = exp_data.get('config', {}).get('model', {})
        
        # Extract model parameters
        backbone = model_config.get('backbone', 'mit_b3')
        num_classes = model_config.get('num_classes', 1)
        img_size = model_config.get('img_size', 352)
        
        # Create model
        model = ColonFormer(
            backbone=backbone,
            num_classes=num_classes,
            img_size=img_size,
            deep_supervision=False,  # No deep supervision for testing
            use_refinement=True
        )
        
        # Load checkpoint
        checkpoint_path = os.path.join(experiment_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded: {backbone} from {checkpoint_path}")
        
        return model, exp_data
    
    def create_test_dataloader(self, dataset_path, img_size=352, batch_size=1):
        """Tạo test dataloader"""
        # Check nếu dataset_path có structure chuẩn
        if os.path.isdir(os.path.join(dataset_path, 'images')):
            # Structure: dataset_path/images/, dataset_path/masks/
            image_dir = os.path.join(dataset_path, 'images')
            mask_dir = os.path.join(dataset_path, 'masks')
        elif os.path.isdir(os.path.join(dataset_path, 'image')):
            # Structure: dataset_path/image/, dataset_path/mask/
            image_dir = os.path.join(dataset_path, 'image')
            mask_dir = os.path.join(dataset_path, 'mask')
        else:
            raise ValueError(f"Invalid dataset structure: {dataset_path}")
        
        # Create dataset
        dataset = PolypDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            img_size=img_size,
            is_training=False  # No augmentation for testing
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"Test dataset loaded: {len(dataset)} samples from {dataset_path}")
        
        return dataloader
    
    def test_model(self, model, test_loader, save_predictions=False, save_dir=None):
        """Test model trên dataset"""
        model.eval()
        
        # Metrics storage
        dice_scores = []
        iou_scores = []
        precisions = []
        recalls = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing', unit='batch')
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = model(images)
                
                # Apply sigmoid and threshold
                predictions = torch.sigmoid(outputs)
                pred_masks = (predictions > 0.5).float()
                
                # Calculate metrics for this batch
                batch_dice = dice_coefficient(pred_masks, masks).cpu().numpy()
                batch_iou = iou_coefficient(pred_masks, masks).cpu().numpy()
                batch_precision, batch_recall = precision_recall(pred_masks, masks)
                batch_precision = batch_precision.cpu().numpy()
                batch_recall = batch_recall.cpu().numpy()
                
                # Store metrics
                dice_scores.extend(batch_dice)
                iou_scores.extend(batch_iou)
                precisions.extend(batch_precision)
                recalls.extend(batch_recall)
                
                # Update progress bar
                current_dice = np.mean(dice_scores)
                current_iou = np.mean(iou_scores)
                pbar.set_postfix({
                    'Dice': f'{current_dice:.4f}',
                    'IoU': f'{current_iou:.4f}'
                })
                
                # Save predictions if requested
                if save_predictions and save_dir:
                    self._save_predictions(
                        images, masks, pred_masks, batch_idx, save_dir
                    )
        
        # Calculate final metrics
        final_metrics = {
            'mDice': np.mean(dice_scores),
            'mIoU': np.mean(iou_scores),
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'std_Dice': np.std(dice_scores),
            'std_IoU': np.std(iou_scores)
        }
        
        return final_metrics
    
    def _save_predictions(self, images, masks, predictions, batch_idx, save_dir):
        """Save prediction visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert to numpy
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        predictions = predictions.cpu().numpy()
        
        for i in range(images.shape[0]):
            # Denormalize image (assuming ImageNet normalization)
            img = images[i].transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            # Get mask and prediction
            mask = (masks[i, 0] * 255).astype(np.uint8)
            pred = (predictions[i, 0] * 255).astype(np.uint8)
            
            # Create visualization
            vis = np.hstack([img, 
                           cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
                           cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)])
            
            # Save
            filename = f'batch_{batch_idx:03d}_sample_{i:03d}.png'
            cv2.imwrite(os.path.join(save_dir, filename), vis)


def test_experiment_on_datasets(experiment_dir, test_datasets, results_manager, 
                               save_predictions=False, force_retest=False):
    """Test một experiment trên multiple datasets"""
    
    # Load experiment info
    exp_data = load_experiment(experiment_dir)
    exp_id = exp_data['experiment_id']
    exp_name = exp_data.get('experiment_name', 'Unknown')
    
    print(f"\nTesting Experiment: {exp_id} ({exp_name})")
    print("-" * 60)
    
    # Check if already tested
    if not force_retest and exp_id in results_manager.results_db:
        print(f"Experiment {exp_id} already tested. Use --force to retest.")
        return
    
    try:
        # Initialize tester và load model
        tester = ModelTester()
        model, exp_data = tester.load_model_from_experiment(experiment_dir)
        
        # Get model image size
        img_size = exp_data.get('config', {}).get('model', {}).get('img_size', 352)
        
        # Test trên từng dataset
        for dataset_name, dataset_path in test_datasets.items():
            if not os.path.exists(dataset_path):
                print(f"Dataset path not found: {dataset_path}, skipping {dataset_name}")
                continue
            
            print(f"\nTesting on {dataset_name}...")
            
            try:
                # Create test dataloader
                test_loader = tester.create_test_dataloader(
                    dataset_path, img_size=img_size, batch_size=4
                )
                
                # Run test
                test_metrics = tester.test_model(model, test_loader)
                
                # Save predictions if requested
                if save_predictions:
                    pred_save_dir = os.path.join(experiment_dir, 'predictions', dataset_name)
                    tester.test_model(model, test_loader, 
                                    save_predictions=True, save_dir=pred_save_dir)
                
                # Add results to manager
                results_manager.add_test_results(
                    experiment_id=exp_id,
                    test_results=test_metrics,
                    dataset_name=dataset_name,
                    test_config={'img_size': img_size, 'batch_size': 4},
                    notes=f"Tested on {len(test_loader.dataset)} samples"
                )
                
                # Print results
                print(f"Results on {dataset_name}:")
                print(f"  mDice: {test_metrics['mDice']:.4f} ± {test_metrics['std_Dice']:.4f}")
                print(f"  mIoU:  {test_metrics['mIoU']:.4f} ± {test_metrics['std_IoU']:.4f}")
                print(f"  Precision: {test_metrics['Precision']:.4f}")
                print(f"  Recall: {test_metrics['Recall']:.4f}")
                
            except Exception as e:
                print(f"Error testing on {dataset_name}: {e}")
                continue
        
        print(f"\nTesting completed for experiment {exp_id}")
        
    except Exception as e:
        print(f"Error testing experiment {exp_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test ColonFormer Experiments')
    
    # Directories
    parser.add_argument('--experiments_dir', type=str, default='checkpoints',
                        help='Directory containing experiments')
    parser.add_argument('--test_data_dir', type=str, default='data/TestDataset',
                        help='Directory containing test datasets')
    
    # Test options
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='Specific experiment ID to test (default: test all untested)')
    parser.add_argument('--datasets', nargs='+', 
                        default=['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir'],
                        help='Datasets to test on')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction visualizations')
    parser.add_argument('--force', action='store_true',
                        help='Force retest even if already tested')
    
    # Actions
    parser.add_argument('--list_experiments', action='store_true',
                        help='List all experiments')
    parser.add_argument('--list_untested', action='store_true',
                        help='List untested experiments')
    parser.add_argument('--summary', action='store_true',
                        help='Show results summary')
    parser.add_argument('--create_paper_table', action='store_true',
                        help='Create paper-style results table')
    
    args = parser.parse_args()
    
    # Initialize results manager
    results_manager = TestResultsManager(args.experiments_dir)
    
    # Prepare test datasets
    test_datasets = {}
    for dataset_name in args.datasets:
        dataset_path = os.path.join(args.test_data_dir, dataset_name)
        if os.path.exists(dataset_path):
            test_datasets[dataset_name] = dataset_path
        else:
            print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
    
    print(f"Available test datasets: {list(test_datasets.keys())}")
    
    # Execute actions
    if args.list_experiments:
        experiments = list_experiments(args.experiments_dir)
        print(f"\nAll Experiments ({len(experiments)}):")
        for exp in experiments:
            status = "COMPLETED" if exp.get('results', {}).get('experiment_completed') else "RUNNING"
            print(f"  {exp['experiment_id']}: {exp.get('experiment_name', 'Unknown')} [{status}]")
    
    elif args.list_untested:
        untested = results_manager.get_untested_experiments()
        print(f"\nUntested Experiments ({len(untested)}):")
        for exp in untested:
            print(f"  {exp['experiment_id']}: {exp.get('experiment_name', 'Unknown')}")
    
    elif args.summary:
        results_manager.print_summary()
    
    elif args.create_paper_table:
        print("\nCreating paper-style results table...")
        paper_table = create_paper_results_table(
            results_manager, 
            output_file=os.path.join(args.experiments_dir, 'paper_results.csv')
        )
        if paper_table is not None:
            print("\nPaper-style Results Table:")
            print(paper_table.to_string(index=False))
    
    else:
        # Run tests
        if args.experiment_id:
            # Test specific experiment
            exp_dir = None
            experiments = list_experiments(args.experiments_dir)
            for exp in experiments:
                if exp['experiment_id'] == args.experiment_id:
                    exp_dir = exp['directory']
                    break
            
            if exp_dir is None:
                print(f"Experiment {args.experiment_id} not found!")
                return
            
            test_experiment_on_datasets(
                exp_dir, test_datasets, results_manager,
                save_predictions=args.save_predictions,
                force_retest=args.force
            )
        else:
            # Test all untested experiments
            untested = results_manager.get_untested_experiments()
            
            if not untested:
                print("No untested experiments found!")
                return
            
            print(f"\nTesting {len(untested)} untested experiments...")
            
            for exp in untested:
                test_experiment_on_datasets(
                    exp['directory'], test_datasets, results_manager,
                    save_predictions=args.save_predictions,
                    force_retest=args.force
                )
        
        # Show summary after testing
        print("\n" + "="*80)
        results_manager.print_summary()


if __name__ == "__main__":
    main() 