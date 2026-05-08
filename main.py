import tkinter as tk
from gui import GestureApp

if __name__ == '__main__':
    root = tk.Tk()
    try:
        GestureApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")  # keeps window open so you can read the error