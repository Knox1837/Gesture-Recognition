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
python main.py 
```
When prompted, type one of the gesture names below and press Enter.
Hold your hand in frame, press **SPACE** to start capturing, **Q** to quit early.

Gestures: `Palm` `Fist` `Thumb` `Index` `Ok` `L` `C` `Down` `Palm_moved` `Fist_moved`

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
The system uses Google mediapipe for hand detection(with 21 unique points) and evaluates it to get gesture info
- Collect all 10 gestures before training
- Checkpoints are saved to `checkpoints/best_model.pth`
- Training curves and confusion matrix are saved to `logs/`

### Inference Pipeline 
1. Capture live webcam frames using OpenCV  
2. Detect hand landmarks using MediaPipe Hands  
3. Extract 21 hand keypoints from the detected hand  
4. Normalize and preprocess landmark coordinates  
5. Convert landmarks into a numerical feature vector  
6. Pass the feature vector into the trained ML model  
7. Predict gesture class and confidence score  
8. Repeat continuously for real-time inference