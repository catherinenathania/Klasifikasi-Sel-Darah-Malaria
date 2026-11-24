import importlib.util
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


ROOT = Path("/mnt/home/rifqicahyono/malaria")
sys.path.insert(0, str(ROOT))
TEST_DIR = ROOT / "malaria_cls_split" / "test"
PILOT_COUNT = 2756


def test_images(limit=PILOT_COUNT):
    return [
        path
        for class_name in ("Parasitized", "Uninfected")
        for path in sorted((TEST_DIR / class_name).glob("*"))
    ][:limit]


def benchmark_yolo(paths):
    torch.set_num_threads(4)
    model = YOLO(str(ROOT / "yolo26n_cls_150ep_val_loss_lr5e4" / "weights" / "best.pt"))
    model.to("cpu")
    for path in paths[:20]:
        model.predict(source=str(path), imgsz=224, batch=1, device="cpu", verbose=False)
    start = time.perf_counter()
    for path in paths:
        model.predict(source=str(path), imgsz=224, batch=1, device="cpu", verbose=False)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_cnn(paths):
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("train_cnn", ROOT / "20_train_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.CustomCNN(num_classes=2)
    checkpoint = torch.load(
        ROOT / "cnn_150ep" / "best_custom_cnn.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    with torch.no_grad():
        for path in paths[:20]:
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0)
            model(image)
        start = time.perf_counter()
        for path in paths:
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0)
            model(image)
        elapsed = time.perf_counter() - start
    return elapsed


def benchmark_svm(paths):
    local = pd.read_csv(ROOT / "malaria_local_features_full.csv")
    global_features = pd.read_csv(ROOT / "malaria_global_features_full.csv")
    manifest = pd.read_csv(ROOT / "malaria_split_manifest.csv")
    manifest = manifest.rename(columns={"class_name": "label"})
    manifest["image_id"] = manifest["source_path"].map(lambda value: Path(value).name)

    test_manifest = manifest[manifest["split"] == "test"].copy()
    test_manifest = test_manifest.merge(
        local[["image_id", "label"] + ["orb_keypoint_count", "orb_mean_response"]],
        on=["image_id", "label"],
        how="left",
    )
    test_manifest = test_manifest.merge(
        global_features[
            ["image_id", "label"]
            + [
                "glcm_contrast",
                "glcm_correlation",
                "glcm_energy",
                "glcm_entropy",
                "shape_area",
                "shape_perimeter",
            ]
        ],
        on=["image_id", "label"],
        how="left",
    )
    feature_columns = [
        "orb_keypoint_count",
        "orb_mean_response",
        "glcm_contrast",
        "glcm_correlation",
        "glcm_energy",
        "glcm_entropy",
        "shape_area",
        "shape_perimeter",
    ]
    features = test_manifest[feature_columns].values[: len(paths)]
    model = joblib.load(ROOT / "svm" / "svm_local_plus_global.joblib")
    model.predict(features[:20])
    start = time.perf_counter()
    model.predict(features)
    elapsed = time.perf_counter() - start
    return elapsed


def summarize(name, elapsed, count):
    per_image = elapsed / count
    return {
        "model": name,
        "images": count,
        "total_seconds": elapsed,
        "milliseconds_per_image": per_image * 1000,
        "images_per_second": count / elapsed,
    }


torch.set_num_threads(4)
torch.set_grad_enabled(False)
paths = test_images(PILOT_COUNT)
results = []
for name, function in (
    ("YOLO26n-cls", benchmark_yolo),
    ("Custom CNN", benchmark_cnn),
    ("SVM Local + Global", benchmark_svm),
):
    elapsed = function(paths)
    results.append(summarize(name, elapsed, len(paths)))

print(platform.platform())
print("CPU cores:", 12)
print("PyTorch threads:", torch.get_num_threads())
for result in results:
    print(result)
