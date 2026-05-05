"""CSV loader, tensor conversion, splits into train/val/test sets"""

import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from config import CSV_PATH, GESTURES, BATCH_SIZE
import numpy as np

class LandmarkDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels  = df['label'].map({g: i for i, g in enumerate(GESTURES)}).values
        self.samples = df.drop(columns=['label']).values.astype('float32')

        # normalize: subtract wrist (landmark 0), scale by hand size
        wrist = self.samples[:, :3]                        # shape (N, 3)
        wrist_tiled = np.tile(wrist, (1, 21))              # shape (N, 63)
        self.samples = self.samples - wrist_tiled
        scale = np.abs(self.samples).max(axis=1, keepdims=True) + 1e-6
        self.samples = self.samples / scale
        

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def get_loaders():
    ds = LandmarkDataset(CSV_PATH)
    n  = len(ds)
    train_n, val_n = int(0.8*n), int(0.1*n)
    test_n = n - train_n - val_n
    train, val, test = random_split(ds, [train_n, val_n, test_n])
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val,   batch_size=BATCH_SIZE),
            DataLoader(test,  batch_size=BATCH_SIZE))