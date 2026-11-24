#!/usr/bin/env python3
"""Visualize labels and predictions for shared test images."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CELL_IMAGES_DIR = BASE_DIR / "cell_images"
SVM_PREDICTIONS = BASE_DIR / "svm" / "svm_test_predictions.csv"
CNN_PREDICTIONS = BASE_DIR / "cnn_150ep" / "cnn_test_predictions.csv"
YOLO_PREDICTIONS = (
    BASE_DIR / "malaria_yolo26n_150ep_val_loss_lr5e4_test_predictions.csv"
)
OUTPUT = BASE_DIR / "laporan" / "visualisasi" / "28_label_vs_prediksi_enam_model.png"


def normalize_id(path_or_id: str) -> str:
    name = Path(path_or_id).name
    return re.sub(r"^\d+_", "", name)


def load_predictions() -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}

    svm = pd.read_csv(SVM_PREDICTIONS)
    for name, frame in svm.groupby("model", sort=False):
        current = frame.copy()
        current["image_id"] = current["image_id"].map(normalize_id)
        predictions[str(name).strip()] = current.set_index("image_id")[["y", "pred"]]

    cnn = pd.read_csv(CNN_PREDICTIONS)
    cnn["image_id"] = cnn["image_path"].map(normalize_id)
    predictions["Custom CNN"] = cnn.set_index("image_id")[["y", "pred"]]

    yolo = pd.read_csv(YOLO_PREDICTIONS)
    label_map = {"Parasitized": 1, "Uninfected": 0}
    yolo["image_id"] = yolo["image_path"].map(normalize_id)
    yolo["y"] = yolo["true_label"].map(label_map)
    yolo["pred"] = yolo["pred_label"].map(label_map)
    predictions["YOLO26n-cls"] = yolo.set_index("image_id")[["y", "pred"]]
    return predictions


def choose_samples(predictions: dict[str, pd.DataFrame]) -> list[str]:
    common_ids = set.intersection(*(set(frame.index) for frame in predictions.values()))
    rows = []
    for image_id in common_ids:
        values = [frame.loc[image_id] for frame in predictions.values()]
        true = int(values[0]["y"])
        errors = sum(int(value["y"]) != int(value["pred"]) for value in values)
        rows.append((true, errors, image_id))

    selected = []
    for true in (1, 0):
        group = sorted(
            (row for row in rows if row[0] == true),
            key=lambda row: (row[1], row[2]),
        )
        selected.append(group[0][2])
        one_error = next((row for row in group if row[1] >= 1), None)
        selected.append(one_error[2] if one_error else group[1][2])
    return selected


def image_path_by_id() -> dict[str, Path]:
    return {
        normalize_id(path.name): path
        for path in CELL_IMAGES_DIR.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }


def main() -> None:
    predictions = load_predictions()
    samples = choose_samples(predictions)
    paths = image_path_by_id()
    preferred_order = [
        "SVM Pixel",
        "SVM Local",
        "SVM Global",
        "SVM Local + Global",
        "Custom CNN",
        "YOLO26n-cls",
    ]
    model_order = [m for m in preferred_order if m in predictions]

    figure, axes = plt.subplots(
        len(model_order),
        len(samples),
        figsize=(14, 3.5 * len(model_order)),
        squeeze=False,
    )

    print("\n=== RINGKASAN CONTOH PREDIKSI PER MODEL ===")
    for row, name in enumerate(model_order):
        print(f"\nModel: {name}")
        for column, image_id in enumerate(samples):
            axis = axes[row, column]
            image = plt.imread(paths[image_id])
            result = predictions[name].loc[image_id]
            true = int(result["y"])
            predicted = int(result["pred"])
            correct = true == predicted
            actual_name = "Parasitized" if true else "Uninfected"
            predicted_name = "Parasitized" if predicted else "Uninfected"
            color = "forestgreen" if correct else "firebrick"
            status_text = "BENAR" if correct else "SALAH"

            print(
                f"  - Sampel {column + 1} ({image_id}): "
                f"Asli={actual_name} | Prediksi={predicted_name} [{status_text}]"
            )

            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

            axis.set_title(
                f"Label: {actual_name}\n"
                f"Prediksi: {predicted_name}\n"
                f"{status_text}",
                color=color,
                fontsize=9,
            )
            if column == 0:
                axis.set_ylabel(name, fontsize=11, fontweight="bold", labelpad=12)

    for column, image_id in enumerate(samples):
        label = "P" if int(predictions["Custom CNN"].loc[image_id]["y"]) else "U"
        axes[0, column].xaxis.set_label_position("top")
        axes[0, column].set_xlabel(
            f"Contoh {label}{column + 1}",
            fontsize=11,
            fontweight="bold",
            labelpad=28,
        )

    figure.suptitle(
        "Label asli dan hasil prediksi pada empat citra uji per model",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Hijau = benar; merah = salah. Setiap kolom menggunakan citra yang sama untuk semua model.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()