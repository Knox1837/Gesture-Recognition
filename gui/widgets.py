"""gui/widgets.py — shared webcam utilities"""

import cv2
from PIL import Image, ImageTk


def frame_to_imgtk(frame, size=(480, 360)):
    """Convert an OpenCV BGR frame to a Tkinter-compatible PhotoImage."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img   = Image.fromarray(rgb).resize(size)
    return ImageTk.PhotoImage(img)


def update_cam_label(label_widget, frame, size=(480, 360)):
    """Flip, resize, and push an OpenCV frame into a tk.Label."""
    frame = cv2.flip(frame, 1)
    imgtk = frame_to_imgtk(frame, size)
    label_widget.configure(image=imgtk)
    label_widget.image = imgtk  # keep reference — prevents garbage collection