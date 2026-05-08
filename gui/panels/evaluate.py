import threading
import tkinter as tk

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageTk

from gui.theme import (
    BG, PANEL, TEXT, TEXT_DIM, BORDER, CHART_BG,
    ACCENT, WARNING, SUCCESS, DANGER,
    FONT_MONO_SM,
    styled_btn, log_box, header, bordered_frame
)

from config import GESTURES


class EvaluatePanel:
    def __init__(self, parent, app):
        self.parent = parent
        self.app    = app

    def build(self):
        header(self.parent, '◎  Evaluate', 'Run the test set and view metrics')

        top = tk.Frame(self.parent, bg=BG)
        top.pack(fill='x', pady=(0, 12))

        self._run_btn = styled_btn(top, '▷  Run Evaluation',
                                   self._run, color=WARNING)
        self._run_btn.pack(side='left')

        split = tk.Frame(self.parent, bg=BG)
        split.pack(fill='both', expand=True)

        self._build_log(split)
        self._build_cm(split)

    # log 

    def _build_log(self, parent):
        log_frame = tk.Frame(parent, bg=BG, width=380)
        log_frame.pack(side='left', fill='y', padx=(0, 10))
        log_frame.pack_propagate(False)

        tk.Label(log_frame, text='Classification report',
                 bg=BG, fg=TEXT_DIM, font=FONT_MONO_SM).pack(anchor='w')
        self._log = log_box(log_frame, height=24)

    # confusion matrix 

    def _build_cm(self, parent):
        cm_frame = bordered_frame(parent)
        cm_frame.pack(side='left', fill='both', expand=True)

        self._cm_lbl = tk.Label(
            cm_frame, bg=PANEL,
            text='Confusion matrix\nwill appear here',
            fg=TEXT_DIM, font=FONT_MONO_SM
        )
        self._cm_lbl.pack(fill='both', expand=True)

    # worker

    def _run(self):
        self._run_btn.configure(state='disabled')
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        import torch
        from src.model   import GestureNet
        from src.dataset import get_loaders
        from src.utils   import load_checkpoint
        from sklearn.metrics import classification_report, confusion_matrix

        def q_log(msg, color=TEXT):
            self.app.log_queue.put((self._log, msg, color))

        try:
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

            report = classification_report(all_labels, all_preds,
                                           target_names=GESTURES)
            q_log(report, TEXT)

            # build confusion matrix image off-screen
            from sklearn.metrics import confusion_matrix as cm_fn
            cm = cm_fn(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(5, 4), facecolor=CHART_BG)
            ax.set_facecolor(CHART_BG)
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(GESTURES)))
            ax.set_yticks(range(len(GESTURES)))
            ax.set_xticklabels(GESTURES, rotation=45, ha='right',
                               fontsize=7, color=TEXT_DIM)
            ax.set_yticklabels(GESTURES, fontsize=7, color=TEXT_DIM)
            for i in range(len(GESTURES)):
                for j in range(len(GESTURES)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=7,
                            color='white' if cm[i, j] > cm.max() / 2 else TEXT_DIM)
            ax.set_title('Confusion Matrix', color=TEXT_DIM, fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            plt.tight_layout()

            canvas_agg = FigureCanvasAgg(fig)
            canvas_agg.draw()
            buf      = canvas_agg.buffer_rgba()
            w, h     = fig.canvas.get_width_height()
            img      = Image.frombytes('RGBA', (w, h), buf)
            imgtk    = ImageTk.PhotoImage(img)
            plt.close(fig)

            def show_cm():
                self._cm_lbl.configure(image=imgtk, text='')
                self._cm_lbl.image = imgtk

            self.app.chart_queue.put(show_cm)
            q_log('\n✓ Evaluation complete.', SUCCESS)

        except Exception as e:
            q_log(f'ERROR: {e}', DANGER)
        finally:
            self.app.chart_queue.put(
                lambda: self._run_btn.configure(state='normal')
            )