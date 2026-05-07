""" Live webcam loop for gesture recognition inference."""
import cv2, torch, numpy as np, mediapipe as mp, os
from model  import GestureNet
from config import CHECKPOINT_DIR, GESTURES
from utils  import load_checkpoint

def run_inference():
    model = GestureNet()
    model = load_checkpoint(model)          # uses utils, consistent with evaluate.py
    model.eval()

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    draw     = mp.solutions.drawing_utils
    cap      = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # mirror for more intuitive interaction

        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label  = "No hand"

        if result.multi_hand_landmarks:
            lm     = result.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten().astype('float32')

            # same normalization as training
            coords -= np.tile(coords[:3], 21)
            coords /= (np.abs(coords).max() + 1e-6)

            with torch.no_grad():
                out  = model(torch.tensor(coords).unsqueeze(0))
                idx  = out.argmax(dim=1).item()
                conf = torch.softmax(out, dim=1).max().item()
                label = f"{GESTURES[idx]}  {conf:.0%}"

        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,100), 2)
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()