"""
Dataset class for ColonFormer training and evaluation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ColonPolypDataset(Dataset):
    """Dataset class for colon polyp segmentation"""
    
    def __init__(self, data_dir, transform=None, phase='train', val_split=0.2, random_state=42):
        """
        Args:
            data_dir: directory containing images and masks folders
            transform: albumentations transform pipeline
            phase: 'train', 'val', or 'all'
            val_split: validation split ratio if phase is 'train' or 'val'
            random_state: random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.phase = phase
        
        # Get image and mask paths
        self.image_paths, self.mask_paths = self._get_paths()
        
        # Split data if needed
        if phase in ['train', 'val']:
            self.image_paths, self.mask_paths = self._split_data(val_split, random_state)
        
        print(f"{phase.capitalize()} dataset: {len(self.image_paths)} samples")
    
    def _get_paths(self):
        """Get image and mask file paths"""
        # Try different folder structures
        possible_image_dirs = ['images', 'image', 'imgs', 'img']
        possible_mask_dirs = ['masks', 'mask', 'labels', 'label', 'gt']
        
        image_dir = None
        mask_dir = None
        
        # Find image directory
        for img_dir in possible_image_dirs:
            candidate = self.data_dir / img_dir
            if candidate.exists():
                image_dir = candidate
                break
        
        # Find mask directory
        for msk_dir in possible_mask_dirs:
            candidate = self.data_dir / msk_dir
            if candidate.exists():
                mask_dir = candidate
                break
        
        if image_dir is None:
            raise ValueError(f"No image directory found in {self.data_dir}")
        if mask_dir is None:
            raise ValueError(f"No mask directory found in {self.data_dir}")
        
        # Get file extensions
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        mask_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        
        # Collect image paths
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(str(image_dir / ext)))
        
        # Collect mask paths
        mask_paths = []
        for ext in mask_extensions:
            mask_paths.extend(glob(str(mask_dir / ext)))
        
        # Sort paths
        image_paths.sort()
        mask_paths.sort()
        
        # Match images and masks
        matched_images = []
        matched_masks = []
        
        for img_path in image_paths:
            img_name = Path(img_path).stem
            
            # Find corresponding mask
            for mask_path in mask_paths:
                mask_name = Path(mask_path).stem
                if img_name == mask_name:
                    matched_images.append(img_path)
                    matched_masks.append(mask_path)
                    break
        
        if len(matched_images) == 0:
            raise ValueError("No matching image-mask pairs found")
        
        print(f"Found {len(matched_images)} image-mask pairs")
        return matched_images, matched_masks
    
    def _split_data(self, val_split, random_state):
        """Split data into train/val"""
        if self.phase == 'train':
            images, _, masks, _ = train_test_split(
                self.image_paths, self.mask_paths,
                test_size=val_split, random_state=random_state, shuffle=True
            )
        elif self.phase == 'val':
            _, images, _, masks = train_test_split(
                self.image_paths, self.mask_paths,
                test_size=val_split, random_state=random_state, shuffle=True
            )
        else:
            images, masks = self.image_paths, self.mask_paths
        
        return images, masks
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Normalize mask to 0-1
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)  # Add channel dimension
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default normalization if no transform
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).permute(2, 0, 1)
        
        return image, mask
    
    def get_sample_info(self, idx):
        """Get information about a sample"""
        return {
            'image_path': self.image_paths[idx],
            'mask_path': self.mask_paths[idx],
            'index': idx
        }


class MultiDatasetLoader(Dataset):
    """Dataset loader that can handle multiple datasets"""
    
    def __init__(self, dataset_configs, transform=None, phase='train'):
        """
        Args:
            dataset_configs: list of dataset configurations
            transform: transform to apply
            phase: train/val/test phase
        """
        self.datasets = []
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        
        total_samples = 0
        
        for config in dataset_configs:
            dataset = ColonPolypDataset(
                data_dir=config['data_dir'],
                transform=transform,
                phase=phase,
                val_split=config.get('val_split', 0.2),
                random_state=config.get('random_state', 42)
            )
            
            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))
            total_samples += len(dataset)
            self.cumulative_lengths.append(total_samples)
        
        print(f"Multi-dataset {phase}: {total_samples} total samples from {len(self.datasets)} datasets")
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        """Get sample from appropriate dataset"""
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths[1:]):
            if idx < cum_len:
                dataset_idx = i
                sample_idx = idx - self.cumulative_lengths[i]
                break
        
        return self.datasets[dataset_idx][sample_idx]


class TestDatasetLoader(Dataset):
    """Dataset loader for test datasets (no masks required)"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: directory containing test images
            transform: transform to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get image paths
        self.image_paths = self._get_image_paths()
        print(f"Test dataset: {len(self.image_paths)} images")
    
    def _get_image_paths(self):
        """Get image file paths"""
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(str(self.data_dir / ext)))
        
        # Also check subdirectories
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                for ext in image_extensions:
                    image_paths.extend(glob(str(subdir / ext)))
        
        image_paths.sort()
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single image"""
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, image_path


def get_default_transforms(img_size=352, phase='train'):
    """Get default transforms for training/validation"""
    
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.3),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def create_dataloaders(config):
    """Create train and validation dataloaders from config"""
    
    # Get transforms
    train_transform = get_default_transforms(config['img_size'], 'train')
    val_transform = get_default_transforms(config['img_size'], 'val')
    
    # Create datasets
    if isinstance(config['data_dir'], list):
        # Multiple datasets
        train_dataset = MultiDatasetLoader(
            config['data_dir'], 
            transform=train_transform, 
            phase='train'
        )
        val_dataset = MultiDatasetLoader(
            config['data_dir'], 
            transform=val_transform, 
            phase='val'
        )
    else:
        # Single dataset
        train_dataset = ColonPolypDataset(
            config['data_dir'], 
            transform=train_transform, 
            phase='train',
            val_split=config.get('val_split', 0.2)
        )
        val_dataset = ColonPolypDataset(
            config['data_dir'], 
            transform=val_transform, 
            phase='val',
            val_split=config.get('val_split', 0.2)
        )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader 