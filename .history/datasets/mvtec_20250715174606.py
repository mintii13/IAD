"""
Dataset loader for Image Anomaly Detection
Supports MVTec, VisA, MPDD, ITDD datasets
Modified from CRAS project
"""

import os
import torch
import PIL
from torch.utils.data import Dataset
from torchvision import transforms
from enum import Enum
import numpy as np


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class IADDataset(Dataset):
    """
    Unified Dataset for Image Anomaly Detection
    Supports MVTec, VisA, MPDD, ITDD
    """
    
    def __init__(
        self,
        config,
        split=DatasetSplit.TRAIN,
        transform=None
    ):
        super().__init__()
        
        self.config = config
        self.split = split
        self.dataset_name = config.DATASET.NAME
        self.root_path = config.DATASET.ROOT
        self.class_names = self._get_class_names()
        self.setting = config.DATASET.SETTING  # 'single' or 'multi'
        
        # Image preprocessing
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        # Mask preprocessing 
        self.mask_transform = transforms.Compose([
            transforms.Resize(config.DATASET.CROP_SIZE),
            transforms.CenterCrop(config.DATASET.CROP_SIZE),
            transforms.ToTensor(),
        ])
        
        # Load dataset
        self.image_paths, self.mask_paths, self.labels = self._load_dataset()
        
    def _get_class_names(self):
        """Get class names based on dataset and setting"""
        if self.setting == 'single':
            # Single class specified in config
            return [self.config.DATASET.CLASS_NAME]
        else:
            # Multi-class: all classes for the dataset
            return self._get_all_classes()
    
    def _get_all_classes(self):
        """Get all class names for each dataset"""
        class_mapping = {
            'mvtec': [
                'bottle', 'cable', 'capsule', 'carpet', 'grid',
                'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
                'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
            ],
            'visa': [
                'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
            ],
            'mpdd': [
                'bracket_black', 'bracket_brown', 'bracket_white',
                'connector', 'metal_plate', 'tubes'
            ],
            'itdd': [
                'cotton_fabric', 'dyed_fabric', 'hemp_fabric', 'plaid_fabric'
            ]
        }
        return class_mapping.get(self.dataset_name, [])
    
    def _get_default_transform(self):
        """Get default image transformations"""
        if self.split == DatasetSplit.TRAIN:
            # Training transforms with augmentation
            return transforms.Compose([
                transforms.Resize(self.config.DATASET.RESIZE),
                transforms.RandomHorizontalFlip(p=self.config.AUG.HORIZONTAL_FLIP),
                transforms.RandomRotation(degrees=self.config.AUG.ROTATION),
                transforms.ColorJitter(
                    brightness=self.config.AUG.COLOR_JITTER,
                    contrast=self.config.AUG.COLOR_JITTER,
                    saturation=self.config.AUG.COLOR_JITTER
                ),
                transforms.CenterCrop(self.config.DATASET.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.AUG.NORMALIZE_MEAN,
                    std=self.config.AUG.NORMALIZE_STD
                ),
            ])
        else:
            # Test transforms without augmentation
            return transforms.Compose([
                transforms.Resize(self.config.DATASET.RESIZE),
                transforms.CenterCrop(self.config.DATASET.CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.AUG.NORMALIZE_MEAN,
                    std=self.config.AUG.NORMALIZE_STD
                ),
            ])
    
    def _load_dataset(self):
        """Load dataset paths and labels"""
        image_paths = []
        mask_paths = []
        labels = []
        
        for class_name in self.class_names:
            class_image_paths, class_mask_paths, class_labels = self._load_class_data(class_name)
            image_paths.extend(class_image_paths)
            mask_paths.extend(class_mask_paths)
            labels.extend(class_labels)
            
        return image_paths, mask_paths, labels
    
    def _load_class_data(self, class_name):
        """Load data for a specific class"""
        image_paths = []
        mask_paths = []
        labels = []
        
        # Determine split folder name
        split_name = self.split.value if self.split != DatasetSplit.VAL else "test"
        
        # Class folder path
        class_path = os.path.join(self.root_path, class_name, split_name)
        ground_truth_path = os.path.join(self.root_path, class_name, "ground_truth")
        
        if not os.path.exists(class_path):
            print(f"Warning: Class path {class_path} does not exist")
            return [], [], []
        
        # Get anomaly types (good, defect types)
        anomaly_types = os.listdir(class_path)
        
        for anomaly_type in sorted(anomaly_types):
            anomaly_path = os.path.join(class_path, anomaly_type)
            if not os.path.isdir(anomaly_path):
                continue
                
            # Get all images in this anomaly type folder
            image_files = sorted([f for f in os.listdir(anomaly_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            
            for image_file in image_files:
                # Image path
                image_path = os.path.join(anomaly_path, image_file)
                image_paths.append(image_path)
                
                # Label (0 for normal, 1 for anomaly)
                label = 0 if anomaly_type == "good" else 1
                labels.append(label)
                
                # Mask path
                if self.split != DatasetSplit.TRAIN and anomaly_type != "good":
                    # Anomaly mask exists for test set
                    mask_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                    mask_path = os.path.join(ground_truth_path, anomaly_type, mask_file)
                    if os.path.exists(mask_path):
                        mask_paths.append(mask_path)
                    else:
                        # Create dummy mask path
                        mask_paths.append(None)
                else:
                    # No mask for normal images or training set
                    mask_paths.append(None)
        
        return image_paths, mask_paths, labels
    
    def __getitem__(self, idx):
        """Get dataset item"""
        # Load image
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Load mask
        mask_path = self.mask_paths[idx]
        if mask_path is not None and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path).convert('L')
            mask = self.mask_transform(mask)
        else:
            # Create dummy mask
            mask = torch.zeros(1, image.shape[1], image.shape[2])
        
        # Get label
        label = self.labels[idx]
        
        return {
            "image": image,
            "mask": mask,
            "label": label,
            "image_path": image_path,
            "mask_path": mask_path if mask_path else "",
        }
    
    def __len__(self):
        return len(self.image_paths)
    
    @property
    def name(self):
        """Dataset name for logging"""
        if self.setting == 'single':
            return f"{self.dataset_name}_{self.class_names[0]}"
        else:
            return f"{self.dataset_name}_multi"


def get_dataloader(config, split, shuffle=None):
    """Get dataloader for specified split"""
    
    # Create dataset
    dataset = IADDataset(config, split=split)
    
    # Determine shuffle
    if shuffle is None:
        shuffle = (split == DatasetSplit.TRAIN)
    
    # Determine batch size
    if split == DatasetSplit.TRAIN:
        batch_size = config.TRAIN.BATCH_SIZE
    else:
        batch_size = config.TEST.BATCH_SIZE
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=(split == DatasetSplit.TRAIN)
    )
    
    return dataloader


def get_dataloaders(config):
    """Get all dataloaders"""
    
    # Training dataloader (only normal images)
    train_loader = get_dataloader(config, DatasetSplit.TRAIN, shuffle=True)
    
    # Test dataloader (normal + anomaly images) 
    test_loader = get_dataloader(config, DatasetSplit.TEST, shuffle=False)
    
    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }
    
    return dataloaders


# Example usage
if __name__ == "__main__":
    # Test dataset loading
    from config.base_config import get_config
    import argparse
    
    # Create dummy args for testing
    args = argparse.Namespace()
    args.dataset = 'mvtec'
    args.data_path = '/path/to/mvtec'
    args.class_name = 'bottle'
    args.setting = 'single'
    args.batch_size = 8
    args.gpu = 0
    args.mem_dim = 2000
    args.use_spatial_memory = True
    args.fusion_method = 'add'
    args.backbone = 'wide_resnet'
    args.epochs = 100
    args.lr = 1e-4
    args.output_dir = './results'
    args.exp_name = 'test'
    
    config = get_config(args)
    
    # Test dataset
    dataset = IADDataset(config, split=DatasetSplit.TRAIN)
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset name: {dataset.name}")
    
    # Test dataloader
    dataloader = get_dataloader(config, DatasetSplit.TRAIN)
    for batch in dataloader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"Labels: {batch['label']}")
        break