from pathlib import Path


SEED = 42
CLASS_NAMES = ("Parasitized", "Uninfected")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
BATCH_SIZE = 32
YOLO_EPOCHS = 150
YOLO_PATIENCE = 10
CNN_EPOCHS = 150
CNN_PATIENCE = 10
WORKING = Path(__file__).resolve().parent

DATASET_DIR = WORKING / "malaria_cls_split"
SPLIT_MANIFEST = WORKING / "malaria_split_manifest.csv"
LOCAL_CSV = WORKING / "malaria_local_features_full.csv"
GLOBAL_CSV = WORKING / "malaria_global_features_full.csv"
CLAHE_DIR = WORKING / "clahe"
CLAHE_ZIP = WORKING / "clahe-20261122T075041Z-1-001.zip"

RUNS_DIR = WORKING
CNN_DIR = WORKING / "cnn_150ep"
SVM_DIR = WORKING / "svm"

