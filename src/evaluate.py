import torch
from model   import GestureNet
from dataset import get_loaders
from utils   import load_checkpoint, plot_confusion_matrix
from config  import GESTURES
from sklearn.metrics import classification_report

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = get_loaders()

    model = GestureNet().to(device)
    model = load_checkpoint(model)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=GESTURES))

    plot_confusion_matrix(all_labels, all_preds)