#!/usr/bin/env python3
"""Visualize misclassified test images (False Positives & False Negatives) across models."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CELL_IMAGES_DIR = BASE_DIR / "cell_images"
SVM_DIR = BASE_DIR / "svm"
CNN_PREDICTIONS = BASE_DIR / "cnn_150ep" / "cnn_test_predictions.csv"
YOLO_PREDICTIONS = (
    BASE_DIR / "malaria_yolo26n_150ep_val_loss_lr5e4_test_predictions.csv"
)
OUTPUT = BASE_DIR / "laporan" / "visualisasi" / "29_sampel_gagal_prediksi.png"


def normalize_id(path_or_id: str) -> str:
    name = Path(path_or_id).name
    return re.sub(r"^\d+_", "", name)


def load_predictions() -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}

    for svm_csv in sorted(SVM_DIR.glob("*predictions*.csv")):
        svm = pd.read_csv(svm_csv)
        if "model" in svm.columns:
            for name, frame in svm.groupby("model", sort=False):
                current = frame.copy()
                id_col = "image_id" if "image_id" in current.columns else "image_path"
                current["image_id"] = current[id_col].map(normalize_id)
                predictions[str(name).strip()] = current.set_index("image_id")[["y", "pred"]]
        else:
            stem = svm_csv.stem.lower()
            if "pixel" in stem:
                name = "SVM Pixel"
            elif "local" in stem and "global" in stem:
                name = "SVM Local + Global"
            elif "local" in stem:
                name = "SVM Local"
            elif "global" in stem:
                name = "SVM Global"
            else:
                name = svm_csv.stem
            id_col = "image_id" if "image_id" in svm.columns else "image_path"
            svm["image_id"] = svm[id_col].map(normalize_id)
            predictions[name] = svm.set_index("image_id")[["y", "pred"]]

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


def choose_failed_samples(
    predictions: dict[str, pd.DataFrame],
    model_order: list[str],
    per_class: int = 2,
) -> list[str]:
    """Memilih sampel yang gagal diprediksi (diurutkan dari jumlah model yang salah terbanyak)."""
    common_ids = set.intersection(*(set(predictions[m].index) for m in model_order))
    rows = []
    for image_id in common_ids:
        true = int(predictions[model_order[0]].loc[image_id, "y"])
        errors = sum(
            int(predictions[m].loc[image_id, "pred"]) != true for m in model_order
        )
        if errors > 0:
            rows.append((true, errors, image_id))

    selected = []
    # Ambil sampel gagal dari kelas Parasitized (False Negative) dan Uninfected (False Positive)
    for true_label in (1, 0):
        group = sorted(
            (r for r in rows if r[0] == true_label),
            key=lambda r: (-r[1], r[2]),  # Prioritaskan yang paling banyak model salahnya
        )
        selected.extend([r[2] for r in group[:per_class]])
    return selected


def image_path_by_id() -> dict[str, Path]:
    return {
        normalize_id(path.name): path
        for path in CELL_IMAGES_DIR.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }


def main() -> None:
    predictions = load_predictions()
    preferred_order = [
        "SVM Pixel",
        "SVM Local",
        "SVM Global",
        "SVM Local + Global",
        "Custom CNN",
        "YOLO26n-cls",
    ]
    model_order = [m for m in preferred_order if m in predictions]
    paths = image_path_by_id()

    # Cetak statistik jumlah gagal prediksi per model di terminal
    common_ids = sorted(set.intersection(*(set(predictions[m].index) for m in model_order)))
    print(f"=== STATISTIK KEGAGALAN PREDIKSI (n_test = {len(common_ids)}) ===")
    for m in model_order:
        df_m = predictions[m].loc[common_ids]
        fn = int(((df_m["y"] == 1) & (df_m["pred"] == 0)).sum())
        fp = int(((df_m["y"] == 0) & (df_m["pred"] == 1)).sum())
        print(f"- {m:<20}: Total Salah = {fn + fp:>3} (FN = {fn:>2}, FP = {fp:>2})")

    # Ambil 4 contoh gagal (2 False Negative / Parasitized, 2 False Positive / Uninfected)
    samples = choose_failed_samples(predictions, model_order, per_class=2)
    if not samples:
        print("Tidak ditemukan sampel yang gagal diprediksi.")
        return

    figure, axes = plt.subplots(
        len(model_order),
        len(samples),
        figsize=(14, 3.3 * len(model_order)),
        squeeze=False,
    )

    for row, name in enumerate(model_order):
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

            if correct:
                status_text = "BENAR"
            else:
                error_type = "FN" if true == 1 else "FP"
                status_text = f"SALAH ({error_type})"

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
        true_val = int(predictions[model_order[0]].loc[image_id, "y"])
        prefix = "FN" if true_val == 1 else "FP"
        axes[0, column].xaxis.set_label_position("top")
        axes[0, column].set_xlabel(
            f"Kasus {prefix}-{column + 1}\n({image_id[:18]}...)",
            fontsize=10,
            fontweight="bold",
            labelpad=28,
        )

    figure.suptitle(
        "Analisis Kesalahan: Contoh Citra Uji yang Gagal Diprediksi per Model",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "FN = False Negative (Parasitized terprediksi Uninfected); FP = False Positive (Uninfected terprediksi Parasitized).",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"\nVisualisasi gagal prediksi disimpan ke: {OUTPUT}")
    print("ID Sampel terpilih:", samples)


if __name__ == "__main__":
    main()