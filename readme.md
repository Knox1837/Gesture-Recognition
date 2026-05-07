# Gesture Recognition System

## Setup

### 1. Venv
```
python -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch (GPU)
```
pip install typing-extensions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Note: This is the backward compatible CUDA 126 download for NVIDIA 3060 laptop GPU\
For CPU installation: ```pip install torch torchvision```

### 3. Install remaining dependencies
```
pip install -r requirements.txt
```

---

## Running

### Collect gesture data (once per gesture for 10 total gestures)
```
python main.py --mode collect
```
When prompted, type one of the gesture names below and press Enter.
Hold your hand in frame, press **SPACE** to start capturing, **Q** to quit early.

Gestures: `palm` `fist` `thumb` `index` `ok` `l` `c` `down` `palm_moved` `fist_moved`

### Train
```
python main.py --mode train
```

### Evaluate
```
python main.py --mode evaluate
```

### Live inference
```
python main.py --mode inference
```
Press **Q** to quit.

---

## Notes
- Collect all 10 gestures before training
- Checkpoints are saved to `checkpoints/best_model.pth`
- Training curves and confusion matrix are saved to `logs/`