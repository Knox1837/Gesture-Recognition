import threading
import tkinter as tk

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.theme import (
    BG, PANEL, ACCENT, ACCENT2, TEXT, TEXT_DIM, BORDER,
    SUCCESS, WARNING, DANGER, CHART_BG,
    FONT_MONO, FONT_MONO_SM,
    styled_btn, log_box, append_log, header, bordered_frame
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import EPOCHS, LR


class TrainPanel:
    def __init__(self, parent, app):
        self.parent     = parent
        self.app        = app
        self._train_losses = []
        self._val_losses   = []
        self._val_accs     = []

    def build(self):
        header(self.parent, '⚡  Train',
               'Train the gesture MLP on collected landmark data')

        # top bar
        top = tk.Frame(self.parent, bg=BG)
        top.pack(fill='x', pady=(0, 12))

        self._start_btn = styled_btn(top, '▷  Start Training',
                                     self._start, color=ACCENT2)
        self._start_btn.pack(side='left', padx=(0, 12))

        self._status_lbl = tk.Label(top, text='', bg=BG, fg=TEXT_DIM,
                                    font=FONT_MONO_SM)
        self._status_lbl.pack(side='left')

        # body: chart left, log right
        split = tk.Frame(self.parent, bg=BG)
        split.pack(fill='both', expand=True)

        self._build_chart(split)
        self._build_log(split)

    # chart

    def _build_chart(self, parent):
        chart_frame = bordered_frame(parent)
        chart_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        self._fig, (self._ax_loss, self._ax_acc) = plt.subplots(
            1, 2, figsize=(6, 3), facecolor=CHART_BG
        )
        self._style_axes()
        plt.tight_layout(pad=1.5)

        self._canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

    def _style_axes(self):
        for ax in (self._ax_loss, self._ax_acc):
            ax.set_facecolor(CHART_BG)
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
        self._ax_loss.set_title('Loss',         color=TEXT_DIM, fontsize=9)
        self._ax_acc.set_title('Val Accuracy',  color=TEXT_DIM, fontsize=9)

    def _redraw_chart(self, tl, vl, va):
        self._ax_loss.clear()
        self._ax_acc.clear()
        self._style_axes()

        self._ax_loss.plot(tl, color=ACCENT2, label='train', linewidth=1.5)
        self._ax_loss.plot(vl, color=WARNING,  label='val',  linewidth=1.5)
        self._ax_loss.legend(fontsize=7, labelcolor=TEXT_DIM,
                             facecolor=PANEL, edgecolor=BORDER)
        self._ax_acc.plot(va, color=ACCENT, linewidth=1.5)
        self._ax_acc.set_ylim(0, 1)

        plt.tight_layout(pad=1.5)
        self._canvas.draw()

    # log 
    def _build_log(self, parent):
        log_frame = tk.Frame(parent, bg=BG, width=280)
        log_frame.pack(side='left', fill='y')
        log_frame.pack_propagate(False)

        tk.Label(log_frame, text='Training log', bg=BG, fg=TEXT_DIM,
                 font=FONT_MONO_SM).pack(anchor='w')
        self._log = log_box(log_frame, height=20)

    # training worker
    def _start(self):
        self._start_btn.configure(state='disabled')
        self._train_losses.clear()
        self._val_losses.clear()
        self._val_accs.clear()
        self._status_lbl.configure(text='Training...', fg=ACCENT)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        import torch
        import torch.nn as nn
        from src.model   import GestureNet
        from src.dataset import get_loaders
        from src.utils   import save_checkpoint

        def q_log(msg, color=TEXT):
            self.app.log_queue.put((self._log, msg, color))

        def q_chart():
            tl = self._train_losses[:]
            vl = self._val_losses[:]
            va = self._val_accs[:]
            self.app.chart_queue.put(lambda: self._redraw_chart(tl, vl, va))

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            q_log(f'Device: {device}', ACCENT)

            train_loader, val_loader, _ = get_loaders()
            model     = GestureNet().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            patience_ctr = 0
            early_stop   = 15

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

                scheduler.step(val_loss)

                self._train_losses.append(train_loss)
                self._val_losses.append(val_loss)
                self._val_accs.append(val_acc)

                color = SUCCESS if val_acc > best_val_acc else TEXT
                q_log(f'Epoch {epoch:3d}/{EPOCHS}  '
                      f'train={train_loss:.4f}  '
                      f'val={val_loss:.4f}  '
                      f'acc={val_acc:.4f}', color)
                q_chart()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_ctr = 0
                    save_checkpoint(model, epoch, val_acc)
                else:
                    patience_ctr += 1
                    if patience_ctr >= early_stop:
                        q_log(f'Early stop at epoch {epoch}', WARNING)
                        break

            q_log(f'\n✓ Done. Best val_acc = {best_val_acc:.4f}', SUCCESS)
            self.app.chart_queue.put(
                lambda: self._status_lbl.configure(
                    text=f'✓ Best acc: {best_val_acc:.4f}', fg=SUCCESS)
            )

        except Exception as e:
            q_log(f'ERROR: {e}', DANGER)
        finally:
            self.app.chart_queue.put(
                lambda: self._start_btn.configure(state='normal')
            )