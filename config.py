DATA_DIR       = r"E:\Projects\Gesture Recognition\data\landmarks"
CSV_PATH       = r"E:\Projects\Gesture Recognition\data\landmarks\gestures.csv"
CHECKPOINT_DIR = r"E:\Projects\Gesture Recognition\models"
LOG_DIR        = r"E:\Projects\Gesture Recognition\logs"

GESTURES = ['palm', 'fist', 'thumb', 'index', 'ok', 'l', 'c', 'down', 'palm_moved', 'fist_moved']

NUM_LANDMARKS  = 21
NUM_COORDS     = 3       # x, y, z per landmark
INPUT_SIZE     = NUM_LANDMARKS * NUM_COORDS   # 63
NUM_CLASSES    = len(GESTURES)                # 10

SAMPLES_PER_GESTURE = 300
CAPTURE_FPS         = 10   # no of samples/sec during collection

BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
HIDDEN_1    = 128
HIDDEN_2    = 64
DROPOUT     = 0.3