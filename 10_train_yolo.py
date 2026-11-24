#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
import shutil

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from ultralytics import YOLO

from common import (
    BATCH_SIZE,
    DATASET_DIR,
    IMAGE_EXTS,
    RUNS_DIR,
    SEED,
    WORKING,
    YOLO_EPOCHS,
    YOLO_PATIENCE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26n-cls")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--run-name", default="yolo26n_cls_150ep_val_loss_lr5e4")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--patience-metric",
        choices=("fitness", "val_loss"),
        default="val_loss",
    )
    parser.add_argument("--val-loss-patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=YOLO_EPOCHS)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    if not DATASET_DIR.exists():
        raise FileNotFoundError("Split belum ada. Jalankan 00_prepare_split.py.")

    if torch.cuda.is_available():
        device = "0"
    else:
        device = "cpu"

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = "yolo26n-cls.pt"
    model = YOLO(model_name)
    custom_val_loss_state = {
        "best_loss": float("inf"),
        "bad_epochs": 0,
        "best_epoch": None,
    }

    if args.patience_metric == "val_loss":
        def on_fit_epoch_end(trainer):
            val_loss = trainer.metrics.get("val/loss")
            if val_loss is None:
                print(
                    "Peringatan: trainer.metrics tidak memiliki val/loss; "
                    "patience validation-loss tidak diaktifkan."
                )
                return

            val_loss = float(val_loss)
            epoch = int(trainer.epoch) + 1
            if val_loss < custom_val_loss_state["best_loss"]:
                custom_val_loss_state["best_loss"] = val_loss
                custom_val_loss_state["bad_epochs"] = 0
                custom_val_loss_state["best_epoch"] = epoch

                weight_dir = RUNS_DIR / args.run_name / "weights"
                epoch_checkpoint = weight_dir / f"epoch{epoch - 1}.pt"
                if epoch_checkpoint.exists():
                    shutil.copy2(
                        epoch_checkpoint,
                        weight_dir / "best.pt",
                    )
                print(
                    f"Val-loss terbaik diperbarui: "
                    f"epoch {epoch}, val/loss {val_loss:.6f}"
                )
            else:
                custom_val_loss_state["bad_epochs"] += 1
                print(
                    f"Patience val-loss: "
                    f"{custom_val_loss_state['bad_epochs']}/"
                    f"{args.val_loss_patience}"
                )
                if (
                    custom_val_loss_state["bad_epochs"]
                    >= args.val_loss_patience
                ):
                    trainer.stop = True
                    print(
                        "Early stopping val-loss: "
                        f"{args.val_loss_patience} epoch "
                        f"tanpa improvement "
                        f"(terbaik epoch "
                        f"{custom_val_loss_state['best_epoch']}, "
                        f"{custom_val_loss_state['best_loss']:.6f})."
                    )

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    model.train(
        data=str(DATASET_DIR),
        epochs=args.epochs,
        imgsz=224,
        batch=BATCH_SIZE,
        device=device,
        workers=2,
        optimizer="AdamW",
        lr0=args.learning_rate,
        patience=(
            0
            if args.patience_metric == "val_loss"
            else YOLO_PATIENCE
        ),
        seed=SEED,
        deterministic=True,
        amp=True,
        plots=True,
        project=str(RUNS_DIR),
        name=args.run_name,
        exist_ok=True,
        save_period=1,
    )

    weight_dir = RUNS_DIR / args.run_name / "weights"
    best_path = weight_dir / "best.pt"
    results_path = RUNS_DIR / args.run_name / "results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        results.columns = [column.strip() for column in results.columns]
        if "val/loss" in results.columns:
            lowest_loss_index = results["val/loss"].idxmin()
            lowest_epoch = int(results.loc[lowest_loss_index, "epoch"])

            epoch_candidates = [
                weight_dir / f"epoch{lowest_epoch}.pt",
                weight_dir / f"epoch{lowest_epoch - 1}.pt",
                weight_dir / f"{lowest_epoch}.pt",
            ]
            epoch_candidate = next(
                (candidate for candidate in epoch_candidates if candidate.exists()),
                None,
            )
            if epoch_candidate is not None:
                shutil.copy2(epoch_candidate, weight_dir / "best.pt")
                print(
                    f"Best YOLO checkpoint dipilih dari validation loss "
                    f"terendah: epoch {lowest_epoch}"
                )
            else:
                print(
                    "Checkpoint epoch validation-loss terendah tidak ditemukan; "
                    "best.pt dari Ultralytics tetap dipakai."
                )
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {best_path}")

    best = YOLO(str(best_path))
    best.val(
        data=str(DATASET_DIR),
        split="test",
        imgsz=224,
        batch=BATCH_SIZE,
        device=device,
        plots=True,
        project=str(RUNS_DIR),
        name="test_eval",
        exist_ok=True,
    )

    idx_to_name = {int(k): str(v) for k, v in best.names.items()}
    name_to_idx = {name: index for index, name in idx_to_name.items()}
    if set(name_to_idx) != {"Parasitized", "Uninfected"}:
        raise RuntimeError(f"Nama kelas tidak cocok: {name_to_idx}")

    test_files = sorted(
        [
            path
            for class_name in ("Parasitized", "Uninfected")
            for path in (DATASET_DIR / "test" / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda path: str(path),
    )

    positive_index = name_to_idx["Parasitized"]
    prediction_rows = []
    for start in range(0, len(test_files), 128):
        paths = test_files[start : start + 128]
        results = best.predict(
            source=[str(path) for path in paths],
            imgsz=224,
            batch=BATCH_SIZE,
            device=device,
            verbose=False,
        )
        for path, result in zip(paths, results):
            prediction_rows.append(
                {
                    "image_path": str(path),
                    "true_label": path.parent.name,
                    "pred_label": idx_to_name[int(result.probs.top1)],
                    "prob_parasitized": float(
                        result.probs.data[positive_index].item()
                    ),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    suffix = args.output_suffix or f"_lr{args.learning_rate:g}"
    prediction_path = WORKING / f"malaria_yolo26n{suffix}_test_predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    y_true = (predictions["true_label"] == "Parasitized").astype(int).values
    y_pred = (predictions["pred_label"] == "Parasitized").astype(int).values
    y_score = predictions["prob_parasitized"].values
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
    )

    metrics = {
        "model": "YOLO26n-cls",
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_parasitized": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_parasitized": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_parasitized": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
    metrics_path = WORKING / f"malaria_yolo26n{suffix}_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Uninfected", "Parasitized"],
            digits=4,
        )
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        matrix, display_labels=["Parasitized", "Uninfected"]
    ).plot(ax=axis, values_format="d")
    axis.set_title("YOLO26n-cls - Confusion Matrix (Test)")
    figure.tight_layout()
    figure.savefig(
        WORKING / f"malaria_yolo26n{suffix}_confusion_matrix.png",
        dpi=180,
    )


if __name__ == "__main__":
    main()

