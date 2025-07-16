"""
Dataset loader for Image Anomaly Detection
Supports MVTec, VisA, MPDD, ITDD datasets
Modified from CRAS project - Following CRAS data processing style
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
    Following CRAS data processing style
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
        self.setting = config.DATASET.SETTING  # 'single' or 'multi'
        
        # CRAS-style parameters
        self.resize = config.DATASET.RESIZE      # 329 (like CRAS)
        self.imgsize = config.DATASET.CROP_SIZE  # 288 (like CRAS)
        self.imagesize = (3, self.imgsize, self.imgsize)  # CRAS-style size info
        
        # Get class names
        self.class_names = self._get_class_names()
        
        # CRAS-style transforms (separate for image and mask)
        self.transform_img = self._get_image_transform()
        self.transform_mask = self._get_mask_transform()
        
        # Override with custom transform if provided
        if transform is not None:
            self.transform_img = transform
        
        # Load dataset paths and labels - CRAS style
        self.imgpaths_per_class, self.data_to_iterate = self._get_image_data()
        
    def _get_class_names(self):
        """Get class names based on dataset and setting"""
        if self.setting == 'single':
            # Single class specified in config (return as list like CRAS)
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
    
    def _get_image_transform(self):
        """Get image transformations - CRAS style but with training augmentation"""
        if self.split == DatasetSplit.TRAIN:
            # Training transforms with augmentation (enhanced from CRAS)
            transform_list = [
                transforms.Resize(self.resize),  # 329 (like CRAS)
                transforms.RandomHorizontalFlip(p=self.config.AUG.HORIZONTAL_FLIP),
                transforms.RandomRotation(degrees=self.config.AUG.ROTATION),
                transforms.ColorJitter(
                    brightness=self.config.AUG.COLOR_JITTER,
                    contrast=self.config.AUG.COLOR_JITTER,
                    saturation=self.config.AUG.COLOR_JITTER
                ),
                transforms.CenterCrop(self.imgsize),  # 288 (like CRAS)
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.AUG.NORMALIZE_MEAN,  # IMAGENET_MEAN
                    std=self.config.AUG.NORMALIZE_STD     # IMAGENET_STD
                ),
            ]
        else:
            # Test transforms without augmentation (exactly like CRAS)
            transform_list = [
                transforms.Resize(self.resize),       # 329
                transforms.CenterCrop(self.imgsize),  # 288
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.AUG.NORMALIZE_MEAN,
                    std=self.config.AUG.NORMALIZE_STD
                ),
            ]
        
        return transforms.Compose(transform_list)
    
    def _get_mask_transform(self):
        """Get mask transformations - exactly like CRAS"""
        transform_list = [
            transforms.Resize(self.resize),       # 329 (like CRAS, not direct resize)
            transforms.CenterCrop(self.imgsize),  # 288 (like CRAS)
            transforms.ToTensor(),
        ]
        return transforms.Compose(transform_list)
    
    def _get_image_data(self):
        """Load image data - CRAS style structure"""
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        
        # Use CRAS-style naming
        set_name = self.split.value if self.split != DatasetSplit.VAL else "test"
        
        for classname in self.class_names:
            classpath = os.path.join(self.root_path, classname, set_name)
            maskpath = os.path.join(self.root_path, classname, "ground_truth")
            
            if not os.path.exists(classpath):
                print(f"Warning: Class path {classpath} does not exist")
                continue
                
            anomaly_types = os.listdir(classpath)
            
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            
            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                if not os.path.isdir(anomaly_path):
                    continue
                    
                anomaly_files = sorted([f for f in os.listdir(anomaly_path) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                imgpaths_per_class[classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]
                
                # Handle masks like CRAS
                if self.split != DatasetSplit.TRAIN and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    if os.path.exists(anomaly_mask_path):
                        anomaly_mask_files = sorted([f for f in os.listdir(anomaly_mask_path)
                                                   if f.lower().endswith('.png')])
                        maskpaths_per_class[classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
                    else:
                        maskpaths_per_class[classname][anomaly] = [None] * len(anomaly_files)
                else:
                    maskpaths_per_class[classname][anomaly] = [None] * len(anomaly_files)
        
        # Create data_to_iterate like CRAS
        data_to_iterate = []
        for classname in self.class_names:
            if classname not in imgpaths_per_class:
                continue
                
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    
                    # Add mask path
                    if self.split != DatasetSplit.TRAIN and anomaly != "good":
                        mask_path = maskpaths_per_class[classname][anomaly][i]
                        data_tuple.append(mask_path)
                    else:
                        data_tuple.append(None)
                    
                    data_to_iterate.append(data_tuple)
        
        return imgpaths_per_class, data_to_iterate
    
    def __getitem__(self, idx):
        """Get dataset item - CRAS style output format"""
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        
        # Load and transform image (like CRAS)
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        
        # Load and transform mask (like CRAS)
        if self.split != DatasetSplit.TRAIN and mask_path is not None and os.path.exists(mask_path):
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            # Create dummy mask with same size as image (like CRAS)
            mask_gt = torch.zeros([1, *image.size()[1:]])
        
        # Return CRAS-style format but compatible with DMIAD
        return {
            "image": image,
            "mask": mask_gt,                              # Use 'mask' (DMIAD style) instead of 'mask_gt'
            "label": int(anomaly != "good"),              # Use 'label' (DMIAD style) instead of 'is_anomaly'
            "image_path": image_path,
            "mask_path": mask_path if mask_path else "",  # Additional info for DMIAD
            "classname": classname,                       # Additional info
            "anomaly_type": anomaly,                      # Additional info
        }
    
    def __len__(self):
        """Dataset length - CRAS style"""
        return len(self.data_to_iterate)
    
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