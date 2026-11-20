"""Shared settings for data prep, training and evaluation.

The web app in app/ does not import this module, so it stays deployable on its
own. It reads the class order from models/model_meta.json, which the training
notebook writes from CLASSES below.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIELD_DIR = DATA_DIR / "field"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# PlantVillage tomato folder names. The final list may shrink after looking at
# the data: classes with too few photos get merged or dropped.
CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

IMAGE_SIZE = 224
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

# Below this top-class probability the app asks for another photo.
# Tune it on the validation set, then copy the value into app/app.py.
CONFIDENCE_THRESHOLD = 0.60
