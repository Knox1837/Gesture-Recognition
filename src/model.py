"""MLP definition for gesture classification, using batch normalization and dropout for regularization"""
import torch.nn as nn
from config import INPUT_SIZE, HIDDEN_1, HIDDEN_2, NUM_CLASSES, DROPOUT

class GestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_1), nn.BatchNorm1d(HIDDEN_1), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),  nn.BatchNorm1d(HIDDEN_2),  nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_2, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x)