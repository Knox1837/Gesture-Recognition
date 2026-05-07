"""gui/app.py — GestureApp: window, sidebar, polling, cam management"""

import queue
import tkinter as tk
from tkinter import ttk

import cv2

from gui.theme import (
    BG, PANEL, SIDEBAR, ACCENT, TEXT, TEXT_DIM, BORDER,
    FONT_MONO, FONT_MONO_SM, FONT_MONO_LG,
)
from gui.panels import CollectPanel, TrainPanel, EvaluatePanel, InferencePanel

PANELS = {
    'collect':   ('⊕  Collect',   CollectPanel),
    'train':     ('⚡  Train',     TrainPanel),
    'evaluate':  ('◎  Evaluate',  EvaluatePanel),
    'inference': ('▷  Inference', InferencePanel),
}


class GestureApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Gesture Recognition')
        self.root.geometry('1100x700')
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        # shared queues (panels post to these; app polls them on main thread)
        self.log_queue   = queue.Queue()
        self.chart_queue = queue.Queue()

        # webcam state (owned here, shared with panels)
        self.cap         = None
        self.cam_running = False

        self.current_mode = None
        self._active_panel = None

        self._build_layout()
        self._show_panel('collect')
        self.root.after(80, self._poll_queues)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # layout

    def _build_layout(self):
        # sidebar
        sidebar = tk.Frame(self.root, bg=SIDEBAR, width=180)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text='◈ GestureAI', bg=SIDEBAR,
            fg=ACCENT, font=FONT_MONO_LG, pady=20
        ).pack(fill='x')

        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', padx=16, pady=4)

        self._nav_btns = {}
        for key, (label, _) in PANELS.items():
            btn = tk.Button(
                sidebar, text=label, anchor='w',
                bg=SIDEBAR, fg=TEXT,
                activebackground=PANEL, activeforeground=ACCENT,
                relief='flat', font=FONT_MONO,
                padx=20, pady=12, cursor='hand2',
                command=lambda k=key: self._show_panel(k)
            )
            btn.pack(fill='x')
            self._nav_btns[key] = btn

        tk.Label(sidebar, text='v1.0', bg=SIDEBAR,
                 fg=TEXT_DIM, font=FONT_MONO_SM).pack(side='bottom', pady=10)

        # main area
        main = tk.Frame(self.root, bg=BG)
        main.pack(side='left', fill='both', expand=True)

        self.panel_frame = tk.Frame(main, bg=BG)
        self.panel_frame.pack(fill='both', expand=True, padx=20, pady=20)

    # panel switching

    def _show_panel(self, mode: str):
        self.stop_cam()

        for w in self.panel_frame.winfo_children():
            w.destroy()

        for key, btn in self._nav_btns.items():
            active = key == mode
            btn.configure(
                bg=PANEL if active else SIDEBAR,
                fg=ACCENT if active else TEXT,
                font=(FONT_MONO[0], FONT_MONO[1], 'bold') if active else FONT_MONO
            )

        self.current_mode  = mode
        _, PanelClass      = PANELS[mode]
        self._active_panel = PanelClass(self.panel_frame, self)
        self._active_panel.build()

    # webcam helpers (called by panels) 

    def start_cam(self):
        self.cap         = cv2.VideoCapture(0)
        self.cam_running = True

    def stop_cam(self):
        self.cam_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    # queue polling (runs on main thread via root.after) 

    def _poll_queues(self):
        # log messages: (ScrolledText widget, message string, color)
        while not self.log_queue.empty():
            widget, msg, _ = self.log_queue.get()
            try:
                if widget.winfo_exists():
                    from gui.theme import append_log
                    append_log(widget, msg)
            except Exception:
                pass

        # chart / UI callables posted by worker threads
        while not self.chart_queue.empty():
            fn = self.chart_queue.get()
            try:
                fn()
            except Exception:
                pass

        self.root.after(80, self._poll_queues)

    # cleanup 

    def _on_close(self):
        self.stop_cam()
        self.root.destroy()