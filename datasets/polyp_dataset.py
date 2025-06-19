"""
Dataset class cho Polyp Segmentation
Hỗ trợ các dataset: Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-T, ETIS-Larib
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
import glob


class PolypDataset(Dataset):
    """
    Dataset class cho polyp segmentation
    """
    def __init__(self, image_paths, mask_paths, img_size=352, transforms=None, mode='train'):
        """
        Args:
            image_paths: list đường dẫn đến images
            mask_paths: list đường dẫn đến masks
            img_size: kích thước ảnh sau resize
            transforms: albumentations transforms
            mode: 'train', 'val', hoặc 'test'
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transforms = transforms
        self.mode = mode
        
        # Đảm bảo số lượng image và mask khớp nhau
        assert len(image_paths) == len(mask_paths), \
            f"Số lượng images ({len(image_paths)}) và masks ({len(mask_paths)}) không khớp!"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image với memory optimization
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask: {mask_path}")
        
        # Optimize memory: use uint8 for processing, float32 only when needed
        mask = (mask > 127).astype(np.uint8)  # Binary threshold on uint8
        
        # Apply transforms
        if self.transforms:
            # Convert mask to float32 only for transforms
            mask_float = mask.astype(np.float32)
            transformed = self.transforms(image=image, mask=mask_float)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Manual basic transforms if no augmentation
            import torch
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Convert to tensors
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).float()
        
        # Ensure mask has correct shape [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask
        }


def get_transforms(img_size=352, mode='train'):
    """
    Tạo transforms cho training và validation theo plan
    """
    if mode == 'train':
        transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),  # scale_limit=0.1 -> scale=(0.9, 1.1) 
                translate_percent=(-0.1, 0.1),  # shift_limit=0.1
                rotate=(-15, 15),  # rotate_limit=15
                shear=(-5, 5),  # Add some shear for more variation
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.Blur(p=0.3),
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.3),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:  # val/test
        transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transforms


class PolypDataModule:
    """
    Data module để quản lý multiple datasets và cross-validation
    """
    def __init__(self, data_root, img_size=352, batch_size=8, 
                 num_workers=4, val_split=0.2, seed=42):
        """
        Args:
            data_root: đường dẫn gốc đến thư mục data
            img_size: kích thước ảnh
            batch_size: batch size
            num_workers: số worker cho DataLoader
            val_split: tỉ lệ validation split
            seed: random seed
        """
        self.data_root = data_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        
        # Check CUDA availability và setup memory optimizations
        import torch
        self.pin_memory = torch.cuda.is_available()
        if not self.pin_memory:
            print("CUDA not available, disabling pin_memory to avoid warnings")
        
        # Optimize num_workers for different platforms
        if num_workers > 0:
            try:
                # Test if multiprocessing works
                import multiprocessing as mp
                mp.set_start_method('spawn', force=True)  # Better for CUDA
            except RuntimeError:
                pass  # Already set
            
            # Adjust num_workers based on system
            import os
            cpu_count = os.cpu_count() or 1
            max_workers = min(cpu_count, 8)  # Cap at 8 để avoid overhead
            
            if num_workers > max_workers:
                print(f"Reducing num_workers from {num_workers} to {max_workers} for better performance")
                self.num_workers = max_workers
            
            # Disable on Windows if problematic
            import platform
            if platform.system() == 'Windows' and self.num_workers > 0:
                print("Windows detected - consider using num_workers=0 if you encounter issues")
        
        # Load data paths
        self.train_images, self.train_masks = [], []
        self.val_images, self.val_masks = [], []
        
        self._load_train_data()
    
    def _load_train_data(self):
        """Load data paths từ TrainDataset"""
        train_path = os.path.join(self.data_root, 'TrainDataset')
        
        if os.path.exists(train_path):
            # Load train data từ TrainDataset folder
            images, masks = self._get_paths_from_train_folder(train_path)
            
            # Split train/val
            from sklearn.model_selection import train_test_split
            train_img, val_img, train_msk, val_msk = train_test_split(
                images, masks, test_size=self.val_split, 
                random_state=self.seed, shuffle=True
            )
            
            self.train_images.extend(train_img)
            self.train_masks.extend(train_msk)
            self.val_images.extend(val_img)
            self.val_masks.extend(val_msk)
            
            print(f"Loaded {len(train_img)} train, {len(val_img)} val samples from TrainDataset")
    
    def _get_paths_from_train_folder(self, train_folder):
        """Get paths từ TrainDataset folder"""
        img_folder = os.path.join(train_folder, 'image')
        mask_folder = os.path.join(train_folder, 'mask')
        
        return self._get_paths_from_folder_structure(img_folder, mask_folder)
    
    def _get_paths_from_folder_structure(self, img_folder, mask_folder):
        """Generic function to get paths from folder structure"""
        if not (os.path.exists(img_folder) and os.path.exists(mask_folder)):
            return [], []
        
        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_paths = []
        for ext in img_extensions:
            image_paths.extend(glob.glob(os.path.join(img_folder, ext)))
        
        mask_paths = []
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # Try to find corresponding mask
            for mask_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                mask_name = base_name + mask_ext
                mask_path = os.path.join(mask_folder, mask_name)
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                    break
        
        min_len = min(len(image_paths), len(mask_paths))
        return sorted(image_paths[:min_len]), sorted(mask_paths[:min_len])
    
    def get_train_dataloader(self):
        """Tạo train dataloader với memory optimizations"""
        transforms = get_transforms(self.img_size, mode='train')
        dataset = PolypDataset(
            self.train_images, self.train_masks, 
            self.img_size, transforms, mode='train'
        )
        
        # Dataloader settings based on num_workers
        dataloader_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': True
        }
        
        # Only add these options if num_workers > 0
        if self.num_workers > 0:
            dataloader_kwargs['persistent_workers'] = True
            dataloader_kwargs['prefetch_factor'] = 2
        
        return DataLoader(**dataloader_kwargs)
    
    def get_val_dataloader(self):
        """Tạo validation dataloader với memory optimizations"""
        transforms = get_transforms(self.img_size, mode='val')
        dataset = PolypDataset(
            self.val_images, self.val_masks,
            self.img_size, transforms, mode='val'
        )
        
        # Dataloader settings based on num_workers
        dataloader_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory
        }
        
        # Only add these options if num_workers > 0
        if self.num_workers > 0:
            dataloader_kwargs['persistent_workers'] = True
            dataloader_kwargs['prefetch_factor'] = 2
        
        return DataLoader(**dataloader_kwargs)
    
    def get_test_dataloader(self, dataset_name):
        """Tạo test dataloader cho dataset cụ thể"""
        dataset_mapping = {
            'Kvasir': 'Kvasir',
            'CVC-ClinicDB': 'CVC-ClinicDB', 
            'CVC-ColonDB': 'CVC-ColonDB',
            'CVC-300': 'CVC-300',
            'CVC-T': 'CVC-300',  # Alias
            'ETIS-Larib': 'ETIS-LaribPolypDB'
        }
        
        folder_name = dataset_mapping.get(dataset_name, dataset_name)
        test_path = os.path.join(self.data_root, 'TestDataset', folder_name)
        
        if not os.path.exists(test_path):
            print(f"Warning: Test dataset {dataset_name} không tồn tại tại {test_path}")
            return None
        
        # Load test data
        images, masks = self._get_paths_from_folder(test_path)
        
        if len(images) == 0:
            print(f"Warning: Không tìm thấy test data cho {dataset_name}")
            return None
        
        transforms = get_transforms(self.img_size, mode='test')
        dataset = PolypDataset(
            images, masks, self.img_size, transforms, mode='test'
        )
        
        return DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
    
    def _get_paths_from_folder(self, folder_path):
        """Get image và mask paths từ test dataset folder"""
        img_folder = os.path.join(folder_path, 'images')
        mask_folder = os.path.join(folder_path, 'masks')
        
        if not (os.path.exists(img_folder) and os.path.exists(mask_folder)):
            return [], []
        
        # Get all image files
        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_paths = []
        for ext in img_extensions:
            image_paths.extend(glob.glob(os.path.join(img_folder, ext)))
        
        # Find corresponding masks
        mask_paths = []
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            # Try different mask extensions
            mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            
            found_mask = False
            for mask_ext in mask_extensions:
                # Remove extension and add mask extension
                base_name = os.path.splitext(img_name)[0]
                mask_name = base_name + mask_ext
                mask_path = os.path.join(mask_folder, mask_name)
                
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                    found_mask = True
                    break
            
            if not found_mask:
                print(f"Warning: Không tìm thấy mask cho {img_path}")
        
        # Ensure same length
        min_len = min(len(image_paths), len(mask_paths))
        return sorted(image_paths[:min_len]), sorted(mask_paths[:min_len])
    
    def print_statistics(self):
        """In thống kê dataset"""
        print("Dataset Statistics:")
        print(f"  Train samples: {len(self.train_images)}")
        print(f"  Val samples: {len(self.val_images)}")
        print(f"  Total samples: {len(self.train_images) + len(self.val_images)}")
        print(f"  Image size: {self.img_size}x{self.img_size}")
        print(f"  Batch size: {self.batch_size}")
        
        print("\nDataLoader Configuration:")
        print(f"  Num workers: {self.num_workers}")
        print(f"  Pin memory: {self.pin_memory}")
        print(f"  CUDA available: {self.pin_memory}")
        
        if len(self.train_images) == 0:
            print("\nWarning: No training data found!")
            print("Make sure your data directory structure is correct:")
            print("  data/")
            print("    TrainDataset/")
            print("      image/")
            print("        *.png, *.jpg, etc.")
            print("      mask/")
            print("        *.png, *.jpg, etc.")


if __name__ == "__main__":
    # Test dataset
    data_root = "data"
    
    data_module = PolypDataModule(
        data_root=data_root,
        img_size=352,
        batch_size=4,
        val_split=0.2
    )
    
    data_module.print_statistics()
    
    # Test dataloaders
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    
    print(f"\nTesting dataloaders...")
    
    # Test train loader
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        masks = batch['mask']
        print(f"Train batch {batch_idx}: images {images.shape}, masks {masks.shape}")
        if batch_idx >= 2:  # Test 3 batches
            break
    
    print("Dataset test completed!") 