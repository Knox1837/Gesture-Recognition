"""webcam capture to save to gestures.csv, using MediaPipe for hand landmark detection"""
import cv2, csv, os, time, mediapipe as mp
from config import CSV_PATH, GESTURES, SAMPLES_PER_GESTURE, CAPTURE_FPS

def run_collection():
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    draw     = mp.solutions.drawing_utils

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(CSV_PATH)

    cap = cv2.VideoCapture(0)
    print("Gestures:", GESTURES)
    gesture = input("Which gesture are you collecting? ").strip()
    assert gesture in GESTURES, f"Unknown gesture: {gesture}"

    collected    = 0
    last_capture = 0
    collecting   = False

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            header = ['label'] + [f'{a}{i}' for i in range(21) for a in ('x','y','z')]
            writer.writerow(header)

        while collected < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1) # mirror for easier collection into a intuitive form
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                now = time.time()
                if collecting and (now - last_capture) >= 1.0 / CAPTURE_FPS:
                    coords = []
                    for p in lm.landmark:
                        coords += [p.x, p.y, p.z]
                    writer.writerow([gesture] + coords)
                    collected    += 1
                    last_capture  = now

            status = f"[SPACE to start] {gesture}: {collected}/{SAMPLES_PER_GESTURE}"
            if collecting:
                status = f"[COLLECTING] {gesture}: {collected}/{SAMPLES_PER_GESTURE}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Collect', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): collecting = True
            if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Collected {collected} samples for '{gesture}'")