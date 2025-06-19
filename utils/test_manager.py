"""
Test Results Manager cho ColonFormer
Auto-detect experiments chưa test và consolidate results
"""

import os
import json
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from .experiment_tracker import load_experiment, list_experiments


class TestResultsManager:
    """
    Quản lý test results cho tất cả experiments
    """
    
    def __init__(self, base_dir, results_file='test_results_summary.json'):
        self.base_dir = base_dir
        self.results_file = os.path.join(base_dir, results_file)
        self.csv_file = os.path.join(base_dir, 'test_results_summary.csv')
        
        # Load existing results
        self.results_db = self._load_results_db()
        
    def _load_results_db(self):
        """Load existing test results database"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        else:
            return OrderedDict()
    
    def _save_results_db(self):
        """Save test results database"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results_db, f, indent=2, default=str)
    
    def get_untested_experiments(self):
        """Get list of experiments chưa được test"""
        all_experiments = list_experiments(self.base_dir)
        untested = []
        
        for exp in all_experiments:
            exp_id = exp['experiment_id']
            if exp_id not in self.results_db:
                # Check nếu experiment đã completed
                if exp.get('results', {}).get('experiment_completed'):
                    untested.append(exp)
        
        return untested
    
    def add_test_results(self, experiment_id, test_results, dataset_name, 
                        test_config=None, notes=None):
        """
        Add test results cho một experiment
        
        Args:
            experiment_id: ID của experiment
            test_results: dict với metrics (mDice, mIoU, Precision, Recall)
            dataset_name: tên dataset được test (Kvasir, CVC-ClinicDB, etc.)
            test_config: config cho test (optional)
            notes: ghi chú thêm (optional)
        """
        if experiment_id not in self.results_db:
            self.results_db[experiment_id] = OrderedDict()
        
        # Test result entry
        test_entry = {
            'dataset': dataset_name,
            'metrics': test_results,
            'test_timestamp': datetime.now().isoformat(),
            'test_config': test_config or {},
            'notes': notes or ""
        }
        
        # Add to database
        if 'test_results' not in self.results_db[experiment_id]:
            self.results_db[experiment_id]['test_results'] = []
        
        self.results_db[experiment_id]['test_results'].append(test_entry)
        
        # Update summary
        self._update_experiment_summary(experiment_id)
        
        # Save to file
        self._save_results_db()
        self._export_to_csv()
        
        print(f"Test results added for experiment {experiment_id} on {dataset_name}")
    
    def _update_experiment_summary(self, experiment_id):
        """Update experiment summary với aggregated results"""
        if experiment_id not in self.results_db:
            return
        
        # Load experiment info
        experiments = list_experiments(self.base_dir)
        exp_data = None
        for exp in experiments:
            if exp['experiment_id'] == experiment_id:
                exp_data = exp
                break
        
        if exp_data is None:
            return
        
        # Extract experiment info
        summary = {
            'experiment_name': exp_data.get('experiment_name', 'Unknown'),
            'experiment_id': experiment_id,
            'timestamp': exp_data.get('config', {}).get('system', {}).get('timestamp', ''),
        }
        
        # Model info
        model_config = exp_data.get('config', {}).get('model', {})
        summary.update({
            'backbone': model_config.get('backbone', 'Unknown'),
            'model_name': model_config.get('name', 'Unknown'),
            'total_parameters': model_config.get('total_parameters', 0),
            'model_size_mb': model_config.get('model_size_mb', 0)
        })
        
        # Training info
        training_config = exp_data.get('config', {}).get('training', {})
        summary.update({
            'epochs': training_config.get('epochs', 0),
            'batch_size': training_config.get('batch_size', 0),
            'learning_rate': training_config.get('lr', 0),
            'optimizer': training_config.get('optimizer', {}).get('type', 'Unknown')
        })
        
        # Training results
        final_results = exp_data.get('results', {}).get('final', {})
        summary.update({
            'best_epoch': final_results.get('best_epoch', 0),
            'best_val_dice': final_results.get('best_val_metrics', {}).get('mDice', 0),
            'final_val_dice': final_results.get('final_val_metrics', {}).get('mDice', 0)
        })
        
        # Test results summary
        test_results = self.results_db[experiment_id].get('test_results', [])
        if test_results:
            # Average across all test datasets
            avg_metrics = self._compute_average_metrics(test_results)
            summary['test_results'] = {
                'num_datasets_tested': len(test_results),
                'avg_test_dice': avg_metrics.get('mDice', 0),
                'avg_test_iou': avg_metrics.get('mIoU', 0),
                'avg_test_precision': avg_metrics.get('Precision', 0),
                'avg_test_recall': avg_metrics.get('Recall', 0)
            }
        else:
            summary['test_results'] = {
                'num_datasets_tested': 0,
                'avg_test_dice': 0,
                'avg_test_iou': 0,
                'avg_test_precision': 0,
                'avg_test_recall': 0
            }
        
        # Update database
        self.results_db[experiment_id]['summary'] = summary
    
    def _compute_average_metrics(self, test_results):
        """Compute average metrics across test datasets"""
        if not test_results:
            return {}
        
        metric_sums = {}
        metric_counts = {}
        
        for result in test_results:
            metrics = result.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1
        
        # Compute averages
        avg_metrics = {}
        for key in metric_sums:
            if metric_counts[key] > 0:
                avg_metrics[key] = metric_sums[key] / metric_counts[key]
        
        return avg_metrics
    
    def _export_to_csv(self):
        """Export results to CSV file for easy viewing"""
        rows = []
        
        for exp_id, exp_data in self.results_db.items():
            summary = exp_data.get('summary', {})
            
            # Base row with experiment info
            base_row = {
                'Experiment_ID': exp_id,
                'Experiment_Name': summary.get('experiment_name', ''),
                'Timestamp': summary.get('timestamp', ''),
                'Backbone': summary.get('backbone', ''),
                'Model_Name': summary.get('model_name', ''),
                'Total_Parameters': summary.get('total_parameters', 0),
                'Model_Size_MB': summary.get('model_size_mb', 0),
                'Epochs': summary.get('epochs', 0),
                'Batch_Size': summary.get('batch_size', 0),
                'Learning_Rate': summary.get('learning_rate', 0),
                'Optimizer': summary.get('optimizer', ''),
                'Best_Epoch': summary.get('best_epoch', 0),
                'Best_Val_Dice': summary.get('best_val_dice', 0),
                'Final_Val_Dice': summary.get('final_val_dice', 0),
            }
            
            # Test results
            test_results = exp_data.get('test_results', [])
            if test_results:
                for test_result in test_results:
                    row = base_row.copy()
                    row.update({
                        'Test_Dataset': test_result.get('dataset', ''),
                        'Test_mDice': test_result.get('metrics', {}).get('mDice', 0),
                        'Test_mIoU': test_result.get('metrics', {}).get('mIoU', 0),
                        'Test_Precision': test_result.get('metrics', {}).get('Precision', 0),
                        'Test_Recall': test_result.get('metrics', {}).get('Recall', 0),
                        'Test_Timestamp': test_result.get('test_timestamp', ''),
                        'Notes': test_result.get('notes', '')
                    })
                    rows.append(row)
            else:
                # No test results yet
                row = base_row.copy()
                row.update({
                    'Test_Dataset': 'NOT_TESTED',
                    'Test_mDice': 0,
                    'Test_mIoU': 0,
                    'Test_Precision': 0,
                    'Test_Recall': 0,
                    'Test_Timestamp': '',
                    'Notes': 'Awaiting test'
                })
                rows.append(row)
        
        # Save to CSV
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values(['Timestamp'], ascending=False)
            df.to_csv(self.csv_file, index=False)
            print(f"Results exported to: {self.csv_file}")
    
    def get_summary_table(self):
        """Get summary table cho tất cả experiments"""
        self._export_to_csv()
        
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            return df
        else:
            return pd.DataFrame()
    
    def print_summary(self):
        """Print summary của tất cả experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        untested = self.get_untested_experiments()
        total_experiments = len(list_experiments(self.base_dir))
        tested_experiments = len(self.results_db)
        
        print(f"Total Experiments: {total_experiments}")
        print(f"Tested Experiments: {tested_experiments}")
        print(f"Untested Experiments: {len(untested)}")
        
        if untested:
            print(f"\nUntested Experiments:")
            for exp in untested:
                print(f"  - {exp['experiment_id']}: {exp.get('experiment_name', 'Unknown')}")
        
        # Top performing experiments
        if self.results_db:
            print(f"\nTop Performing Experiments (by average test Dice):")
            sorted_experiments = []
            for exp_id, exp_data in self.results_db.items():
                summary = exp_data.get('summary', {})
                test_summary = summary.get('test_results', {})
                avg_dice = test_summary.get('avg_test_dice', 0)
                if avg_dice > 0:
                    sorted_experiments.append((exp_id, summary, avg_dice))
            
            sorted_experiments.sort(key=lambda x: x[2], reverse=True)
            
            for i, (exp_id, summary, avg_dice) in enumerate(sorted_experiments[:5]):
                backbone = summary.get('backbone', 'Unknown')
                num_datasets = summary.get('test_results', {}).get('num_datasets_tested', 0)
                print(f"  {i+1}. {exp_id} ({backbone}): {avg_dice:.4f} Dice ({num_datasets} datasets)")
        
        print(f"\nResults files:")
        print(f"  JSON: {self.results_file}")
        print(f"  CSV: {self.csv_file}")
        print("="*80)
    
    def get_experiment_by_id(self, experiment_id):
        """Get experiment info theo ID"""
        experiments = list_experiments(self.base_dir)
        for exp in experiments:
            if exp['experiment_id'] == experiment_id:
                return exp
        return None


def create_paper_results_table(results_manager, output_file=None):
    """
    Tạo bảng kết quả theo format của paper (như bảng trong ảnh)
    """
    df = results_manager.get_summary_table()
    
    if df.empty:
        print("No test results available")
        return None
    
    # Group by method (backbone)
    methods = df['Backbone'].unique()
    datasets = df[df['Test_Dataset'] != 'NOT_TESTED']['Test_Dataset'].unique()
    
    # Create paper-style table
    paper_table = []
    
    for method in methods:
        method_data = df[df['Backbone'] == method]
        
        # Get best experiment for this method
        best_exp = method_data.loc[method_data['Best_Val_Dice'].idxmax()]
        
        row = {
            'Method': f"ColonFormer-{method}",
            'Parameters': f"{best_exp['Total_Parameters']:,}",
            'Model_Size_MB': f"{best_exp['Model_Size_MB']:.1f}"
        }
        
        # Add results for each dataset
        for dataset in datasets:
            dataset_results = method_data[method_data['Test_Dataset'] == dataset]
            if not dataset_results.empty:
                dice = dataset_results['Test_mDice'].values[0]
                iou = dataset_results['Test_mIoU'].values[0]
                row[f'{dataset}_mDice'] = f"{dice:.3f}"
                row[f'{dataset}_mIoU'] = f"{iou:.3f}"
            else:
                row[f'{dataset}_mDice'] = "N/A"
                row[f'{dataset}_mIoU'] = "N/A"
        
        paper_table.append(row)
    
    paper_df = pd.DataFrame(paper_table)
    
    if output_file:
        paper_df.to_csv(output_file, index=False)
        print(f"Paper-style results table saved to: {output_file}")
    
    return paper_df 