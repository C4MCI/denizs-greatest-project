"""
PyTorch Dataset & DataLoader utilities
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class SensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 mean=None, std=None, fit_stats=False):
        """
        X : (N, T, C)  float32
        y : (N,)       int64
        """
        self.X = X.copy()
        self.y = y.copy()

        if fit_stats:
            # compute stats over (N, T) per channel
            self.mean = self.X.mean(axis=(0, 1), keepdims=True)  # (1,1,C)
            self.std  = self.X.std(axis=(0, 1), keepdims=True) + 1e-8
        else:
            self.mean = mean
            self.std  = std

        if self.mean is not None:
            self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long))

    @property
    def class_weights(self):
        """For WeightedRandomSampler to handle class imbalance."""
        counts  = np.bincount(self.y)
        weights = 1.0 / counts
        return torch.tensor(weights[self.y], dtype=torch.float32)


def make_loaders(data_dir='data', batch_size=64, num_workers=0):
    X_tr = np.load(f'{data_dir}/train_X.npy')
    y_tr = np.load(f'{data_dir}/train_y.npy')
    X_va = np.load(f'{data_dir}/val_X.npy')
    y_va = np.load(f'{data_dir}/val_y.npy')
    X_te = np.load(f'{data_dir}/test_X.npy')
    y_te = np.load(f'{data_dir}/test_y.npy')

    train_ds = SensorDataset(X_tr, y_tr, fit_stats=True)
    val_ds   = SensorDataset(X_va, y_va, mean=train_ds.mean, std=train_ds.std)
    test_ds  = SensorDataset(X_te, y_te, mean=train_ds.mean, std=train_ds.std)

    # oversample minority class (failures are rare)
    sampler = WeightedRandomSampler(train_ds.class_weights,
                                    num_samples=len(train_ds),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds
