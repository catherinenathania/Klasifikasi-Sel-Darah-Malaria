#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import (
    CNN_DIR,
    RUNS_DIR,
    SPLIT_MANIFEST,
    SVM_DIR,
    WORKING,
)


def metrics(y_true, predicted, scores):
    return {
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def main() -> None:
    import matplotlib.pyplot as plt

    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError("Manifest belum ada.")

    manifest = pd.read_csv(SPLIT_MANIFEST)
    manifest["image_id"] = manifest["source_path"].map(
        lambda path: Path(path).name
    )
    path_to_id = dict(zip(manifest["image_path"], manifest["image_id"]))
    predictions = {}

    yolo_path = WORKING / "malaria_yolo26n_test_predictions.csv"
    if yolo_path.exists():
        frame = pd.read_csv(yolo_path)
        predictions["YOLO26n-cls"] = pd.DataFrame(
            {
                "image_id": frame["image_path"].map(path_to_id),
                "y": (frame["true_label"] == "Parasitized").astype(int),
                "pred": (frame["pred_label"] == "Parasitized").astype(int),
                "score": frame["prob_parasitized"],
            }
        )

    cnn_path = CNN_DIR / "cnn_test_predictions.csv"
    if cnn_path.exists():
        frame = pd.read_csv(cnn_path)
        predictions["Custom CNN"] = pd.DataFrame(
            {
                "image_id": frame["image_path"].map(path_to_id),
                "y": frame["y"],
                "pred": frame["pred"],
                "score": frame["score"],
            }
        )

    svm_path = SVM_DIR / "svm_test_predictions.csv"
    if svm_path.exists():
        frame = pd.read_csv(svm_path)
        for name, group in frame.groupby("model", sort=False):
            predictions[name] = group[
                ["image_id", "y", "pred", "score"]
            ].reset_index(drop=True)

    if not predictions:
        raise RuntimeError(
            "Belum ada prediksi. Jalankan YOLO/CNN/SVM terlebih dahulu."
        )
    for name, frame in predictions.items():
        if frame["image_id"].isna().any():
            raise RuntimeError(f"image_id tidak dapat dipetakan untuk {name}")

    order = [
        "YOLO26n-cls",
        "Custom CNN",
        "SVM Pixel",
        "SVM Local",
        "SVM Global",
        "SVM Local + Global",
    ]
    names = [name for name in order if name in predictions]

    table_a = pd.DataFrame(
        [
            {
                "model": name,
                **metrics(
                    predictions[name]["y"],
                    predictions[name]["pred"],
                    predictions[name]["score"],
                ),
            }
            for name in names
        ]
    )

    common_ids = set.intersection(
        *[set(predictions[name]["image_id"]) for name in names]
    )
    table_b = pd.DataFrame(
        [
            {
                "model": name,
                **metrics(
                    predictions[name].loc[
                        predictions[name]["image_id"].isin(common_ids), "y"
                    ],
                    predictions[name].loc[
                        predictions[name]["image_id"].isin(common_ids), "pred"
                    ],
                    predictions[name].loc[
                        predictions[name]["image_id"].isin(common_ids), "score"
                    ],
                ),
            }
            for name in names
        ]
    )

    table_a.to_csv(WORKING / "comparison_A_own_test.csv", index=False)
    table_b.to_csv(WORKING / "comparison_B_common_test.csv", index=False)
    print("TABEL A - test set masing-masing model")
    print(table_a.round(4).to_string(index=False))
    print(f"\nTABEL B - test set bersama (n={len(common_ids):,})")
    print(table_b.round(4).to_string(index=False))

    axes = table_b.set_index("model")[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].plot(kind="bar", figsize=(12, 5), rot=20)
    axes.set_ylim(0.5, 1.0)
    axes.set_title(
        f"Perbandingan model pada test set yang sama (n={len(common_ids):,})"
    )
    axes.grid(axis="y")
    plt.tight_layout()
    plt.savefig(WORKING / "comparison_all_models.png", dpi=300)

    cnn_history = CNN_DIR / "training_history.csv"
    yolo_history = RUNS_DIR / "yolo26n_cls_150ep_val_loss_lr5e4" / "results.csv"
    curves = {}
    if cnn_history.exists():
        frame = pd.read_csv(cnn_history)
        curves["Custom CNN"] = pd.DataFrame(
            {
                "epoch": frame["epoch"],
                "train_loss": frame["train_loss"],
                "val_loss": frame["val_loss"],
                "train_acc": frame["train_accuracy"],
                "val_acc": frame["val_accuracy"],
            }
        )
    if yolo_history.exists():
        frame = pd.read_csv(yolo_history)
        frame.columns = [column.strip() for column in frame.columns]
        curves["YOLO26n-cls"] = pd.DataFrame(
            {
                "epoch": frame["epoch"],
                "train_loss": frame["train/loss"],
                "val_loss": frame["val/loss"],
                "train_acc": pd.NA,
                "val_acc": frame["metrics/accuracy_top1"],
            }
        )

    if curves:
        figure, axes = plt.subplots(
            2,
            len(curves),
            figsize=(7 * len(curves), 8),
            squeeze=False,
        )
        for column, (name, frame) in enumerate(curves.items()):
            best_epoch = int(frame.loc[frame["val_loss"].idxmin(), "epoch"])
            axis = axes[0, column]
            axis.plot(frame["epoch"], frame["train_loss"], label="Train loss")
            axis.plot(frame["epoch"], frame["val_loss"], label="Val loss")
            axis.axvline(
                best_epoch,
                color="gray",
                linestyle="--",
                label=f"Val loss terendah (epoch {best_epoch})",
            )
            axis.set_title(f"{name} - Loss")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.grid(True)
            axis.legend()

            axis = axes[1, column]
            if frame["train_acc"].notna().any():
                axis.plot(frame["epoch"], frame["train_acc"], label="Train acc")
            axis.plot(frame["epoch"], frame["val_acc"], label="Val acc")
            axis.set_title(f"{name} - Accuracy")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Accuracy")
            axis.set_ylim(0.90, 1.0)
            axis.grid(True)
            axis.legend()

        figure.tight_layout()
        figure.savefig(
            WORKING / "training_curves_all_models.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()

