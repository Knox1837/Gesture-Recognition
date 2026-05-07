import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='Gesture Recognition System')
    parser.add_argument('--mode',
                        choices=['collect', 'train', 'evaluate', 'inference', 'gui'],
                        default='gui',
                        help='Which mode to run (default: gui)')
    args = parser.parse_args()

    if args.mode == 'gui':
        import tkinter as tk
        from gui.app import GestureApp
        root = tk.Tk()
        GestureApp(root)
        root.mainloop()
    elif args.mode == 'collect':
        from src.collect import run_collection
        run_collection()
    elif args.mode == 'train':
        from src.train import run_training
        run_training()
    elif args.mode == 'evaluate':
        from src.evaluate import run_evaluation
        run_evaluation()
    elif args.mode == 'inference':
        from src.inference import run_inference
        run_inference()

if __name__ == '__main__':
    main()