"""Central configuration for the Fake News Bi-LSTM project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Paths ----
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
GLOVE_PATH = ROOT / "glove" / "glove.6B.100d.txt"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "bilstm_fake_news.keras"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# ---- Data ----
# WELFake's Kaggle page says 0 = fake, 1 = real, but the CSV is actually
# 0 = real, 1 = fake (Reuters articles are all label 0). So no flip is needed:
# the model predicts P(fake) with 1 = FAKE, 0 = REAL.
FLIP_LABELS = False
TEST_SIZE = 0.10
VAL_SIZE = 0.10
SEED = 42

# ---- Text vectorization ----
MAX_TOKENS = 50_000     # vocabulary size
MAX_LEN = 300           # tokens per article (pad / truncate)

# ---- Model ----
EMBED_DIM = 100         # must match the GloVe file (glove.6B.100d)
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT = 0.3
SPATIAL_DROPOUT = 0.2
TRAIN_EMBEDDINGS = False  # freeze GloVe first; set True to fine-tune

# ---- Training ----
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 2
