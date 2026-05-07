import csv
import os
import time
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

from gui.theme import (
    BG, PANEL, ACCENT, BORDER, TEXT_DIM, SUCCESS, WARNING, DANGER,
    FONT_MONO, FONT_MONO_SM, FONT_MONO_LG,
    styled_btn, log_box, header, bordered_frame
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import GESTURES, CSV_PATH, SAMPLES_PER_GESTURE, CAPTURE_FPS


class CollectPanel:
    def __init__(self, parent, app):
        """
        parent : tk.Frame  — the cleared panel_frame from app.py
        app    : GestureApp — gives access to cap, cam_running, root.after
        """
        self.parent = parent
        self.app    = app

        self._collecting      = False
        self._count           = 0
        self._gesture_name    = ''
        self._last_capture_t  = 0
        self._collect_file    = None
        self._collect_writer  = None

        self._mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7)
        self._mp_draw  = mp.solutions.drawing_utils

    def build(self):
        header(self.parent, '⊕  Collect',
               'Capture hand landmark samples for each gesture')

        content = tk.Frame(self.parent, bg=BG)
        content.pack(fill='both', expand=True)

        self._build_controls(content)
        self._build_feed(content)

        self.app.start_cam()
        self._loop()

    # controls (left column)

    def _build_controls(self, parent):
        left = tk.Frame(parent, bg=BG, width=260)
        left.pack(side='left', fill='y', padx=(0, 16))
        left.pack_propagate(False)

        tk.Label(left, text='Gesture', bg=BG, fg=TEXT_DIM,
                 font=FONT_MONO_SM).pack(anchor='w')

        self._gesture_var = tk.StringVar(value=GESTURES[0])
        ttk.Combobox(
            left, textvariable=self._gesture_var,
            values=GESTURES, state='readonly', width=22,
            font=FONT_MONO
        ).pack(fill='x', pady=(2, 12))

        tk.Label(left, text='Progress', bg=BG, fg=TEXT_DIM,
                 font=FONT_MONO_SM).pack(anchor='w')

        self._progress_var = tk.IntVar(value=0)
        self._progress_lbl = tk.Label(
            left, text=f'0 / {SAMPLES_PER_GESTURE}',
            bg=BG, fg=ACCENT, font=FONT_MONO_LG
        )
        self._progress_lbl.pack(anchor='w', pady=(2, 4))

        ttk.Progressbar(
            left, maximum=SAMPLES_PER_GESTURE,
            variable=self._progress_var, length=220
        ).pack(fill='x', pady=(0, 16))

        self._status_lbl = tk.Label(
            left, text='● Ready', bg=BG, fg=TEXT_DIM, font=FONT_MONO
        )
        self._status_lbl.pack(anchor='w', pady=(0, 12))

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill='x')

        self._start_btn = styled_btn(btn_row, '▷  Start', self._start,
                                     color=ACCENT, width=10)
        self._start_btn.pack(side='left', padx=(0, 6))

        self._stop_btn = styled_btn(btn_row, '■  Stop', self._stop,
                                    color=DANGER, width=8)
        self._stop_btn.pack(side='left')
        self._stop_btn.configure(state='disabled')

        tk.Label(
            left,
            text='Tip: hold gesture steady,\npress Start to begin capture.',
            bg=BG, fg=TEXT_DIM, font=FONT_MONO_SM, justify='left'
        ).pack(anchor='w', pady=(16, 0))

    # webcam feed (right column) 

    def _build_feed(self, parent):
        right = bordered_frame(parent)
        right.pack(side='left', fill='both', expand=True)

        self._cam_lbl = tk.Label(right, bg='#000')
        self._cam_lbl.pack(fill='both', expand=True)

    # camera loop

    def _loop(self):
        if not self.app.cam_running or self.app.current_mode != 'collect':
            return

        ret, frame = self.app.cap.read() if self.app.cap else (False, None)
        if ret:
            frame_flip = cv2.flip(frame, 1)
            rgb        = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
            result     = self._mp_hands.process(rgb)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                self._mp_draw.draw_landmarks(
                    frame_flip, lm, mp.solutions.hands.HAND_CONNECTIONS)
                if self._collecting:
                    self._save_landmark(lm)

            status_txt = 'CAPTURING' if self._collecting else 'READY'
            color_cv   = (0, 212, 170) if self._collecting else (107, 114, 128)
            cv2.putText(frame_flip, status_txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_cv, 2)
            cv2.putText(frame_flip, f'{self._count}/{SAMPLES_PER_GESTURE}',
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_cv, 1)

            imgtk = ImageTk.PhotoImage(
                Image.fromarray(
                    cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
                ).resize((480, 360))
            )
            self._cam_lbl.configure(image=imgtk)
            self._cam_lbl.image = imgtk

        self.app.root.after(30, self._loop)

    # capture logic 

    def _start(self):
        self._gesture_name   = self._gesture_var.get()
        self._count          = 0
        self._last_capture_t = 0
        self._collecting     = True
        self._progress_var.set(0)
        self._progress_lbl.configure(text=f'0 / {SAMPLES_PER_GESTURE}')

        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        write_header      = not os.path.exists(CSV_PATH)
        self._collect_file   = open(CSV_PATH, 'a', newline='')
        self._collect_writer = csv.writer(self._collect_file)
        if write_header:
            header_row = ['label'] + [
                f'{a}{i}' for i in range(21) for a in ('x', 'y', 'z')
            ]
            self._collect_writer.writerow(header_row)

        self._status_lbl.configure(text='● Capturing...', fg=ACCENT)
        self._start_btn.configure(state='disabled')
        self._stop_btn.configure(state='normal')

    def _save_landmark(self, hand_landmarks):
        now = time.time()
        if now - self._last_capture_t < 1.0 / CAPTURE_FPS:
            return
        if self._count >= SAMPLES_PER_GESTURE:
            self._finish()
            return

        coords = []
        for p in hand_landmarks.landmark:
            coords += [p.x, p.y, p.z]
        self._collect_writer.writerow([self._gesture_name] + coords)
        self._count         += 1
        self._last_capture_t = now

        self._progress_var.set(self._count)
        self._progress_lbl.configure(text=f'{self._count} / {SAMPLES_PER_GESTURE}')

        if self._count >= SAMPLES_PER_GESTURE:
            self._finish()

    def _finish(self):
        self._collecting = False
        if self._collect_file:
            self._collect_file.close()
        self._status_lbl.configure(
            text=f'✓ Done — {self._gesture_name}', fg=SUCCESS)
        self._start_btn.configure(state='normal')
        self._stop_btn.configure(state='disabled')

    def _stop(self):
        self._collecting = False
        if self._collect_file:
            self._collect_file.close()
        self._status_lbl.configure(text='■ Stopped', fg=WARNING)
        self._start_btn.configure(state='normal')
        self._stop_btn.configure(state='disabled')