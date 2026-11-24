#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hinge_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from common import (
    GLOBAL_CSV,
    LOCAL_CSV,
    SEED,
    SPLIT_MANIFEST,
    SVM_DIR,
)


LOCAL_COLS = ["orb_keypoint_count", "orb_mean_response"]
GLOBAL_COLS = [
    "glcm_contrast",
    "glcm_correlation",
    "glcm_energy",
    "glcm_entropy",
    "shape_area",
    "shape_perimeter",
]


def load_pixels(frame: pd.DataFrame, size: int) -> np.ndarray:
    output = np.zeros((len(frame), size * size * 3), dtype=np.float32)
    for index, (_, row) in enumerate(
        tqdm(frame.iterrows(), total=len(frame), desc="Membaca gambar")
    ):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            image_path = Path(row["source_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Gambar tidak ditemukan: {row['image_path']}")

        with Image.open(image_path) as image:
            output[index] = (
                np.asarray(
                    image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ).ravel()
                / 255.0
            )
    return output


def make_svm(
    c_value: float,
    gamma_multiplier: float,
    feature_count: int,
    use_pca: bool,
):
    steps = [StandardScaler()]
    effective_count = feature_count
    if use_pca:
        effective_count = min(100, feature_count)
        steps.append(PCA(n_components=effective_count, random_state=SEED))
    steps.append(
        SVC(
            kernel="rbf",
            C=c_value,
            gamma=gamma_multiplier / effective_count,
            class_weight="balanced",
            random_state=SEED,
        )
    )
    return make_pipeline(*steps)


def evaluate(model, features, labels):
    scores = model.decision_function(features)
    predictions = (scores > 0).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
    }
    return metrics, predictions, scores


def make_gpu_svm(c_value, gamma_multiplier, feature_count, train_features, train_labels):
    import cupy as cp
    from cuml.decomposition import PCA
    from cuml.preprocessing import StandardScaler
    from cuml.svm import SVC

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(cp.asarray(train_features, dtype=cp.float32))
    effective_count = feature_count
    pca = PCA(n_components=min(100, feature_count))
    effective_count = min(100, feature_count)
    scaled_train = pca.fit_transform(scaled_train)

    model = SVC(
        kernel="rbf",
        C=c_value,
        gamma=gamma_multiplier / effective_count,
        class_weight="balanced",
    )
    model.fit(scaled_train, cp.asarray(train_labels))
    return model, scaler, pca


def transform_gpu(features, scaler, pca):
    import cupy as cp

    scaled = scaler.transform(cp.asarray(features, dtype=cp.float32))
    return pca.transform(scaled)


def evaluate_gpu(model, features, labels, scaler, pca):
    import cupy as cp

    transformed = transform_gpu(features, scaler, pca)
    scores = cp.asnumpy(model.decision_function(transformed))
    predictions = (scores > 0).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
    }
    return metrics, predictions, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate SVM ablation")
    parser.add_argument("--pixel-size", type=int, default=32)
    parser.add_argument("--global-csv", type=Path, default=GLOBAL_CSV)
    parser.add_argument("--local-csv", type=Path, default=LOCAL_CSV)
    parser.add_argument("--grid-subsample", type=int, default=8000)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("pixel", "local", "global", "local_global"),
        default=("pixel", "local", "global", "local_global"),
        help="Empat representasi SVM: Pixel, Local, Global, atau Local + Global.",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="CPU memakai scikit-learn; GPU memakai RAPIDS cuML.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError("Manifest belum ada. Jalankan 00_prepare_split.py.")
    if not args.local_csv.exists():
        raise FileNotFoundError(f"CSV fitur lokal tidak ditemukan: {args.local_csv}")

    SVM_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(SPLIT_MANIFEST).rename(columns={"class_name": "label"})
    manifest["image_id"] = manifest["source_path"].map(
        lambda path: Path(path).name
    )
    local = pd.read_csv(args.local_csv)

    frames = [(manifest, "manifest"), (local, "local")]
    for frame, name in frames:
        duplicates = int(frame.duplicated(["image_id", "label"]).sum())
        if duplicates:
            print(f"PERINGATAN: {duplicates} duplikat di {name}; baris pertama dipakai.")

    manifest = manifest.drop_duplicates(["image_id", "label"])
    local = local.drop_duplicates(["image_id", "label"])

    merged = manifest.merge(
        local[["image_id", "label"] + LOCAL_COLS],
        on=["image_id", "label"],
        how="left",
    )
    merged["y"] = (merged["label"] == "Parasitized").astype(int)
    merged["has_local"] = merged[LOCAL_COLS].notna().all(axis=1)

    global_available = args.global_csv.exists()
    if global_available:
        global_frame = pd.read_csv(args.global_csv)
        duplicates = int(global_frame.duplicated(["image_id", "label"]).sum())
        if duplicates:
            print(
                f"PERINGATAN: {duplicates} duplikat di global; baris pertama dipakai."
            )
        global_frame = global_frame.drop_duplicates(["image_id", "label"])
        merged = merged.merge(
            global_frame[["image_id", "label"] + GLOBAL_COLS],
            on=["image_id", "label"],
            how="left",
        )
        merged["has_global"] = merged[GLOBAL_COLS].notna().all(axis=1)
        if set(args.experiments) & {"local", "global", "local_global"}:
            keep = merged["has_local"] & merged["has_global"]
            print("Mode fitur: memakai subset lokal + global yang sama.")
        else:
            keep = merged["has_local"]
    else:
        if "global" in args.experiments:
            raise FileNotFoundError(
                f"Mode global membutuhkan CSV: {args.global_csv}"
            )
        print("Fitur global belum tersedia; hanya SVM dengan fitur lokal dapat dijalankan.")
        keep = merged["has_local"]

    frame = merged[keep].reset_index(drop=True)
    print(f"Baris SVM: {len(frame):,}")
    print(pd.crosstab(frame["split"], frame["label"]))
    for split_name in ("train", "val", "test"):
        if (frame["split"] == split_name).sum() == 0:
            raise RuntimeError(f"Split '{split_name}' kosong setelah penggabungan.")

    pixel_cache = SVM_DIR / f"pixels_{args.pixel_size}_n{len(frame)}.npy"
    if pixel_cache.exists():
        pixel_features = np.load(pixel_cache)
    else:
        pixel_features = load_pixels(frame, args.pixel_size)
        np.save(pixel_cache, pixel_features)

    representations = {}
    if "local" in args.experiments:
        representations["SVM Local"] = frame[LOCAL_COLS].values.astype(np.float64)
    if "global" in args.experiments:
        representations["SVM Global"] = frame[GLOBAL_COLS].values.astype(np.float64)
    if "local_global" in args.experiments:
        representations["SVM Local + Global"] = np.concatenate(
            [frame[LOCAL_COLS].values, frame[GLOBAL_COLS].values],
            axis=1,
        ).astype(np.float64)
    if "pixel" in args.experiments:
        representations["SVM Pixel"] = pixel_features.astype(np.float64)

    rng = np.random.RandomState(SEED)
    train_mask = (frame["split"] == "train").values
    val_mask = (frame["split"] == "val").values
    test_mask = (frame["split"] == "test").values
    results = {}
    grid_rows = []

    for name, features in representations.items():
        train_features = features[train_mask]
        train_labels = frame.loc[train_mask, "y"].values
        val_features = features[val_mask]
        val_labels = frame.loc[val_mask, "y"].values
        test_features = features[test_mask]
        test_labels = frame.loc[test_mask, "y"].values

        if args.grid_subsample and len(train_features) > args.grid_subsample:
            indices = rng.choice(
                len(train_features), args.grid_subsample, replace=False
            )
        else:
            indices = np.arange(len(train_features))

        started = time.time()
        best = None
        for c_value in (0.1, 1, 10, 100):
            for gamma_multiplier in (0.1, 1, 10):
                model_path = SVM_DIR / f"{name.lower().replace(' ', '_').replace('+','plus')}.joblib"
                if args.backend == "gpu":
                    import cupy as cp

                    cp.cuda.runtime.setDevice(args.gpu_index)
                    model, scaler, pca = make_gpu_svm(
                        c_value,
                        gamma_multiplier,
                        features.shape[1],
                        train_features[indices],
                        train_labels[indices],
                    )
                    val_scores = cp.asnumpy(
                        model.decision_function(
                            transform_gpu(val_features, scaler, pca)
                        )
                    )
                else:
                    model = make_svm(
                        c_value,
                        gamma_multiplier,
                        features.shape[1],
                        use_pca=True,
                    ).fit(train_features[indices], train_labels[indices])
                    val_scores = model.decision_function(val_features)
                signed_val_labels = np.where(val_labels == 1, 1, -1)
                val_loss = hinge_loss(signed_val_labels, val_scores)
                val_predictions = (val_scores > 0).astype(int)
                val_f1 = f1_score(val_labels, val_predictions, zero_division=0)
                grid_rows.append(
                    {
                        "model": name,
                        "C": c_value,
                        "gamma_mult": gamma_multiplier,
                        "val_loss": val_loss,
                        "val_f1": val_f1,
                    }
                )
                if best is None or val_loss < best["val_loss"]:
                    best = {
                        "C": c_value,
                        "gamma_mult": gamma_multiplier,
                        "val_loss": val_loss,
                    }

        if args.backend == "gpu":
            final_model, final_scaler, final_pca = make_gpu_svm(
                best["C"],
                best["gamma_mult"],
                features.shape[1],
                train_features,
                train_labels,
            )
            try:
                joblib.dump(
                    {
                        "model": final_model,
                        "scaler": final_scaler,
                        "pca": final_pca,
                    },
                    model_path,
                )
            except Exception as exc:
                print(f"GPU SVM checkpoint tidak dapat disimpan: {exc}")
            val_metrics, _, _ = evaluate_gpu(
                final_model,
                val_features,
                val_labels,
                final_scaler,
                final_pca,
            )
            test_metrics, predictions, scores = evaluate_gpu(
                final_model,
                test_features,
                test_labels,
                final_scaler,
                final_pca,
            )
        else:
            final_model = make_svm(
                best["C"],
                best["gamma_mult"],
                features.shape[1],
                use_pca=True,
            ).fit(train_features, train_labels)
            try:
                joblib.dump(final_model, model_path)
            except Exception as exc:
                print(f"CPU SVM checkpoint tidak dapat disimpan: {exc}")
            val_metrics, _, _ = evaluate(
                final_model, val_features, val_labels
            )
            test_metrics, predictions, scores = evaluate(
                final_model, test_features, test_labels
            )
        print(
            f"{name:<28} C={best['C']:<5} gamma={best['gamma_mult']:<4} "
            f"val_loss={best['val_loss']:.4f} | "
            f"test acc={test_metrics['accuracy']:.4f} "
            f"F1={test_metrics['f1']:.4f} "
            f"AUC={test_metrics['roc_auc']:.4f} "
            f"({time.time() - started:.0f}s)"
        )
        results[name] = {
            "best": best,
            "val": val_metrics,
            "test": test_metrics,
            "y_test": test_labels,
            "pred": predictions,
            "score": scores,
        }

    result_rows = []
    prediction_rows = []
    test_ids = frame.loc[test_mask, "image_id"].values
    baseline_f1 = None
    if "SVM Pixel" in results:
        baseline_f1 = results["SVM Pixel"]["test"]["f1"]
    if baseline_f1 is None and (SVM_DIR / "svm_results.csv").exists():
        existing_results = pd.read_csv(SVM_DIR / "svm_results.csv")
        baseline_rows = existing_results[
            existing_results["model"] == "SVM Pixel"
        ]
        if not baseline_rows.empty:
            baseline_f1 = float(baseline_rows.iloc[0]["test_f1"])
    if baseline_f1 is None:
        baseline_f1 = 0.0
    ablation_rows = []

    for name, result in results.items():
        result_rows.append(
            {
                "model": name,
                "n_fitur": representations[name].shape[1],
                "C": result["best"]["C"],
                "gamma_mult": result["best"]["gamma_mult"],
                "val_loss": result["best"]["val_loss"],
                **{f"test_{key}": value for key, value in result["test"].items()},
            }
        )
        ablation_rows.append(
            {
                "Model": name,
                "Test Accuracy": result["test"]["accuracy"],
                "Test Precision": result["test"]["precision"],
                "Test Recall": result["test"]["recall"],
                "Test F1-Score": result["test"]["f1"],
                "Delta F1": result["test"]["f1"] - baseline_f1,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "image_id": test_ids,
                    "y": result["y_test"],
                    "pred": result["pred"],
                    "score": result["score"],
                }
            )
        )

    new_results = pd.DataFrame(result_rows)
    if (SVM_DIR / "svm_results.csv").exists():
        old_results = pd.read_csv(SVM_DIR / "svm_results.csv")
        names = set(new_results["model"])
        old_results = old_results[~old_results["model"].isin(names)]
        combined_results = pd.concat([old_results, new_results], ignore_index=True)
    else:
        combined_results = new_results
    combined_results.to_csv(SVM_DIR / "svm_results.csv", index=False)

    new_predictions = pd.concat(prediction_rows)
    if (SVM_DIR / "svm_test_predictions.csv").exists():
        old_predictions = pd.read_csv(SVM_DIR / "svm_test_predictions.csv")
        names = set(new_predictions["model"])
        old_predictions = old_predictions[
            ~old_predictions["model"].isin(names)
        ]
        old_predictions = old_predictions.dropna(axis=1, how="all")
        new_predictions = new_predictions.dropna(axis=1, how="all")
        combined_predictions = pd.concat(
            [old_predictions, new_predictions], ignore_index=True
        )
    else:
        combined_predictions = new_predictions
    combined_predictions.to_csv(
        SVM_DIR / "svm_test_predictions.csv", index=False
    )
    ablation = pd.DataFrame(ablation_rows)
    ablation_path = SVM_DIR / "svm_ablation_results.csv"
    if ablation_path.exists():
        old_ablation = pd.read_csv(ablation_path)
        names = set(ablation["Model"])
        old_ablation = old_ablation[~old_ablation["Model"].isin(names)]
        ablation = pd.concat([old_ablation, ablation], ignore_index=True)
    ablation.to_csv(ablation_path, index=False)
    print(ablation.round(4).to_string(index=False))

    grid_frame = pd.DataFrame(grid_rows)
    grid_path = SVM_DIR / "svm_validation_grid.csv"
    grid_frame.to_csv(grid_path, index=False)

    unique_models = grid_frame["model"].unique()
    figure, axes = plt.subplots(
        1,
        len(unique_models),
        figsize=(6 * len(unique_models), 5),
        squeeze=False,
    )
    for axis, name in zip(axes[0], unique_models):
        subset = grid_frame[grid_frame["model"] == name]
        heatmap = subset.pivot(
            index="C",
            columns="gamma_mult",
            values="val_loss",
        )
        sns.heatmap(
            heatmap,
            annot=True,
            fmt=".4f",
            cmap="viridis_r",
            ax=axis,
            cbar_kws={"label": "Validation hinge loss"},
        )
        axis.set_title(f"{name} - Validation Grid")
        axis.set_xlabel("Gamma multiplier")
        axis.set_ylabel("C")
    figure.tight_layout()
    figure.savefig(SVM_DIR / "svm_validation_grid_loss.png", dpi=200)

    figure, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
    for axis, (name, result) in zip(axes, results.items()):
        matrix = confusion_matrix(
            result["y_test"], result["pred"], labels=[1, 0]
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
            xticklabels=["Parasitized", "Uninfected"],
            yticklabels=["Parasitized", "Uninfected"],
        )
        axis.set_title(name)
    figure.tight_layout()
    figure.savefig(SVM_DIR / "svm_confusion_matrices.png", dpi=200)

    plt.figure(figsize=(7, 6))
    for name, result in results.items():
        false_positive, true_positive, _ = roc_curve(
            result["y_test"], result["score"]
        )
        plt.plot(
            false_positive,
            true_positive,
            label=f"{name} (AUC={result['test']['roc_auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - SVM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SVM_DIR / "svm_roc.png", dpi=200)


if __name__ == "__main__":
    main()

