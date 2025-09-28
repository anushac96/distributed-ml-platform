import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import logging
from typing import Iterator, Optional, List
import asyncio
import re
import math

logger = logging.getLogger(__name__)

class DistributedSampler(Sampler):
    """Custom sampler that splits data across workers"""
    
    def __init__(self, dataset, worker_id: str, num_workers: int, shuffle: bool = True):
        self.dataset = dataset

        # Ensure num_workers is a sane int
        try:
            self.num_workers = max(1, int(num_workers))
        except Exception:
            self.num_workers = 1

        # Parse worker_id robustly:
        # - trailing digits => use numeric suffix
        # - trailing hex (UUID-like) => use hex prefix
        # - fallback to hashed value mod num_workers
        worker_index = 0
        try:
            if worker_id is None:
                worker_id = "0"
            s = str(worker_id)
            m = re.search(r'(\d+)$', s)
            if m:
                worker_index = int(m.group(1)) % self.num_workers
            else:
                m2 = re.search(r'([0-9a-fA-F]{1,16})$', s)
                if m2:
                    worker_index = int(m2.group(1)[:8], 16) % self.num_workers
                else:
                    worker_index = abs(hash(s)) % self.num_workers
        except Exception:
            worker_index = 0

        self.worker_id = int(worker_index)
        self.shuffle = shuffle
        self.epoch = 0
        
        # Calculate data split for this worker
        self.total_size = len(dataset)

        # Use ceil split so last worker gets remainder; avoid per_worker == 0
        self.per_worker = max(1, math.ceil(self.total_size / self.num_workers)) if self.total_size > 0 else 0
        self.start_idx = min(self.worker_id * self.per_worker, self.total_size)
        self.end_idx = min((self.worker_id + 1) * self.per_worker, self.total_size)
         
        logger.info(f"Worker {worker_id} (index={self.worker_id}) assigned data indices {self.start_idx}:{self.end_idx}")

    def __iter__(self) -> Iterator[int]:
        indices = list(range(self.start_idx, self.end_idx))
         
        if self.shuffle:
            # Use worker_id as seed for consistent but different shuffling
            np.random.seed(self.epoch + self.worker_id)
            np.random.shuffle(indices)
         
        return iter(indices)
    
    def __len__(self) -> int:
        return max(0, self.end_idx - self.start_idx)
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch

class MNISTDataModule:
    """MNIST dataset with distributed loading"""
    
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.train_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Download MNIST dataset"""
        try:
            torchvision.datasets.MNIST(
                self.data_dir, 
                train=True, 
                download=True,
                transform=self.transform
            )
            torchvision.datasets.MNIST(
                self.data_dir, 
                train=False, 
                download=True,
                transform=self.transform
            )
            logger.info("MNIST dataset prepared successfully")
        except Exception as e:
            logger.error(f"Error preparing MNIST data: {e}")
    
    def get_train_dataloader(self, worker_id: str, num_workers: int) -> DataLoader:
        """Get training dataloader for specific worker"""
        if self.train_dataset is None:
            self.train_dataset = torchvision.datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.transform
            )
        
        sampler = DistributedSampler(
            self.train_dataset, 
            worker_id, 
            num_workers,
            shuffle=True
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,  # DataLoader worker threads
            pin_memory=torch.cuda.is_available()
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test dataloader (not distributed)"""
        if self.test_dataset is None:
            self.test_dataset = torchvision.datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.transform
            )
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False  # Disable pin_memory
        )

class CIFAR10DataModule:
    """CIFAR-10 dataset for more complex training"""
    
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # More sophisticated transforms for CIFAR-10
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.train_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Download CIFAR-10 dataset"""
        try:
            torchvision.datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=self.train_transform
            )
            torchvision.datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                transform=self.test_transform
            )
            logger.info("CIFAR-10 dataset prepared successfully")
        except Exception as e:
            logger.error(f"Error preparing CIFAR-10 data: {e}")
    
    def get_train_dataloader(self, worker_id: str, num_workers: int) -> DataLoader:
        """Get training dataloader for specific worker"""
        if self.train_dataset is None:
            self.train_dataset = torchvision.datasets.CIFAR10(
                self.data_dir,
                train=True,
                transform=self.train_transform
            )
        
        sampler = DistributedSampler(
            self.train_dataset,
            worker_id,
            num_workers,
            shuffle=True
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=2,  # DataLoader worker threads
            pin_memory=torch.cuda.is_available()  # Only pin memory if GPU available
        )