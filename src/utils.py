import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import CHECKPOINT_DIR, LOG_DIR, GESTURES

def save_checkpoint(model, epoch, val_acc, filename='best_model.pth'):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        'epoch':     epoch,
        'model_state_dict': model.state_dict(),
        'val_acc':   val_acc,
    }, path)
    print(f"  Checkpoint saved → {path}  (val_acc={val_acc:.4f})")

def load_checkpoint(model, filename='best_model.pth'):
    path = os.path.join(CHECKPOINT_DIR, filename)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}  (val_acc={checkpoint['val_acc']:.4f})")
    return model

def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """Re-center on wrist (landmark 0) and scale by max extent."""
    coords = coords.copy()
    wrist  = coords[:3]
    coords -= np.tile(wrist, 21)
    scale  = np.abs(coords).max() + 1e-6
    return coords / scale

def plot_training_curves(train_losses, val_losses, val_accs):
    os.makedirs(LOG_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train loss')
    ax1.plot(val_losses,   label='Val loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True)

    ax2.plot(val_accs, label='Val accuracy', color='green')
    ax2.set_title('Validation accuracy'); ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(LOG_DIR, 'training_curves.png')
    plt.savefig(path)
    plt.show()
    print(f"Training curves saved → {path}")

def plot_confusion_matrix(all_labels, all_preds):
    os.makedirs(LOG_DIR, exist_ok=True)
    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=GESTURES)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title('Confusion matrix')

    plt.tight_layout()
    path = os.path.join(LOG_DIR, 'confusion_matrix.png')
    plt.savefig(path)
    plt.show()
    print(f"Confusion matrix saved → {path}")