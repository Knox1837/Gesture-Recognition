import tkinter as tk

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk

from gui.theme import (
    BG, PANEL, BORDER, ACCENT, SUCCESS, DANGER, TEXT_DIM,
    FONT_MONO_PRED, FONT_MONO,
    styled_btn, header, bordered_frame
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import GESTURES


class InferencePanel:
    def __init__(self, parent, app):
        self.parent  = parent
        self.app     = app
        self._running = False
        self._model   = None

    def build(self):
        header(self.parent, '▷  Inference',
               'Live gesture recognition from webcam')

        top = tk.Frame(self.parent, bg=BG)
        top.pack(fill='x', pady=(0, 12))

        self._start_btn = styled_btn(top, '▷  Start', self._start,
                                     color=SUCCESS, width=10)
        self._start_btn.pack(side='left', padx=(0, 8))

        self._stop_btn = styled_btn(top, '■  Stop', self._stop,
                                    color=DANGER, width=8)
        self._stop_btn.pack(side='left')
        self._stop_btn.configure(state='disabled')

        self._pred_lbl = tk.Label(top, text='—', bg=BG, fg=ACCENT,
                                  font=FONT_MONO_PRED)
        self._pred_lbl.pack(side='left', padx=24)

        self._conf_lbl = tk.Label(top, text='', bg=BG, fg=TEXT_DIM,
                                  font=FONT_MONO)
        self._conf_lbl.pack(side='left')

        cam_frame = bordered_frame(self.parent)
        cam_frame.pack(fill='both', expand=True)

        self._cam_lbl = tk.Label(cam_frame, bg='#000')
        self._cam_lbl.pack(fill='both', expand=True)

    # start / stop

    def _start(self):
        from src.model import GestureNet
        from src.utils import load_checkpoint

        try:
            self._model = GestureNet()
            self._model = load_checkpoint(self._model)
            self._model.eval()
        except Exception as e:
            self._pred_lbl.configure(text='Error', fg=DANGER)
            self._conf_lbl.configure(text=str(e))
            return

        self._mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7)
        self._mp_draw  = mp.solutions.drawing_utils

        self.app.start_cam()
        self._running = True
        self._start_btn.configure(state='disabled')
        self._stop_btn.configure(state='normal')
        self._loop()

    def _stop(self):
        self._running = False
        self.app.stop_cam()
        self._pred_lbl.configure(text='—')
        self._conf_lbl.configure(text='')
        self._start_btn.configure(state='normal')
        self._stop_btn.configure(state='disabled')

    # inference loop

    def _loop(self):
        import torch

        if not self._running or self.app.current_mode != 'inference':
            return

        ret, frame = self.app.cap.read() if self.app.cap else (False, None)
        if ret:
            frame_flip = cv2.flip(frame, 1)
            rgb        = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
            result     = self._mp_hands.process(rgb)
            label, conf_txt = '—', ''

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                self._mp_draw.draw_landmarks(
                    frame_flip, lm, mp.solutions.hands.HAND_CONNECTIONS)

                coords = np.array(
                    [[p.x, p.y, p.z] for p in lm.landmark]
                ).flatten().astype('float32')
                coords -= np.tile(coords[:3], 21)
                coords /= (np.abs(coords).max() + 1e-6)

                with torch.no_grad():
                    out  = self._model(torch.tensor(coords).unsqueeze(0))
                    idx  = out.argmax(dim=1).item()
                    conf = torch.softmax(out, dim=1).max().item()
                    label    = GESTURES[idx]
                    conf_txt = f'{conf:.0%}'

                cv2.putText(frame_flip, f'{label} {conf_txt}', (10, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 212, 170), 2)

            self._pred_lbl.configure(text=label)
            self._conf_lbl.configure(text=conf_txt)

            imgtk = ImageTk.PhotoImage(
                Image.fromarray(
                    cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
                ).resize((640, 420))
            )
            self._cam_lbl.configure(image=imgtk)
            self._cam_lbl.image = imgtk

        self.app.root.after(30, self._loop)