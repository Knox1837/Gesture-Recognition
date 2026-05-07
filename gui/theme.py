"""gui/theme.py — colour palette, fonts, shared widget factories"""

import tkinter as tk
from tkinter import ttk, scrolledtext

# palette 
BG       = '#0f1117'
PANEL    = '#1a1d27'
SIDEBAR  = '#13151f'
ACCENT   = '#00d4aa'
ACCENT2  = '#7c6af7'
TEXT     = '#e8eaf0'
TEXT_DIM = '#6b7280'
BORDER   = '#2a2d3a'
SUCCESS  = '#22c55e'
WARNING  = '#f59e0b'
DANGER   = '#ef4444'
CHART_BG = '#1a1d27'

FONT_MONO      = ('Courier', 11)
FONT_MONO_SM   = ('Courier', 9)
FONT_MONO_LG   = ('Courier', 13, 'bold')
FONT_MONO_HERO = ('Courier', 18, 'bold')
FONT_MONO_PRED = ('Courier', 20, 'bold')

# widget factories 

def styled_btn(parent, text, command, color=ACCENT, width=18):
    return tk.Button(
        parent, text=text, command=command,
        bg=color, fg=BG,
        activebackground=TEXT, activeforeground=BG,
        relief='flat', font=('Courier', 11, 'bold'),
        padx=14, pady=8, cursor='hand2', width=width
    )


def log_box(parent, height=10):
    box = scrolledtext.ScrolledText(
        parent, bg=PANEL, fg=TEXT, insertbackground=TEXT,
        font=FONT_MONO_SM, relief='flat',
        height=height, state='disabled',
        selectbackground=ACCENT2
    )
    box.pack(fill='both', expand=True, pady=(8, 0))
    return box


def append_log(box: scrolledtext.ScrolledText, msg: str):
    box.configure(state='normal')
    box.insert('end', msg + '\n')
    box.see('end')
    box.configure(state='disabled')


def header(parent, title: str, subtitle: str = ''):
    tk.Label(
        parent, text=title, bg=BG, fg=TEXT,
        font=FONT_MONO_HERO, anchor='w'
    ).pack(fill='x', pady=(0, 2))
    if subtitle:
        tk.Label(
            parent, text=subtitle, bg=BG, fg=TEXT_DIM,
            font=FONT_MONO_SM, anchor='w'
        ).pack(fill='x', pady=(0, 12))
    ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=(0, 16))


def bordered_frame(parent):
    return tk.Frame(
        parent, bg=PANEL, bd=0,
        highlightthickness=1, highlightbackground=BORDER
    )