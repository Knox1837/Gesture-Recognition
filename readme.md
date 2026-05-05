# Install dependencies 
- pip install torch torchvision mediapipe opencv-python pandas numpy scikit-learn matplotlib

# Collect data — run once per gesture (10 times total)
- python main.py --mode collect
type: palm → press SPACE → hold gesture → repeat for all 10

# Train
- python main.py --mode train

# Evaluate
- python main.py --mode evaluate

# Live demo
- python main.py --mode inference