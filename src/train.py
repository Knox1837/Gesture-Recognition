""" Training loop for gesture recognition model """
import torch
import torch.nn as nn
from model   import GestureNet
from dataset import get_loaders
from utils   import save_checkpoint, plot_training_curves
from config  import EPOCHS, LR, CHECKPOINT_DIR

def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    train_loader, val_loader, _ = get_loaders()

    model     = GestureNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc  = 0.0
    patience_ctr  = 0
    early_stop    = 15          # stop if no improvement for 15 epochs

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # validate
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y  = x.to(device), y.to(device)
                out   = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                correct  += (out.argmax(dim=1) == y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc   = correct / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

        # checkpoint & early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            save_checkpoint(model, epoch, val_acc)
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stop} epochs)")
                break

    print(f"\nBest val_acc: {best_val_acc:.4f}")
    plot_training_curves(train_losses, val_losses, val_accs)