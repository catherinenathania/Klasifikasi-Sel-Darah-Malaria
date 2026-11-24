from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --- KONFIGURASI ---
BASE_DIR = Path(__file__).resolve().parent
# Ubah DATASET_DIR ke folder tempat gambar sel malaria (Parasitized/ & Uninfected/) tersimpan di laptop Anda
DATASET_DIR = BASE_DIR / "dataset"  
OUTPUT_PATH = BASE_DIR / "laporan" / "visualisasi" / "28_label_vs_prediksi_enam_model.png"

def label_str(val):
    return "Parasitized" if int(val) == 1 else "Uninfected"

def find_image_file(image_id: str, raw_path: str) -> Path | None:
    # 1. Cek jika path asli di CSV langsung ada
    p = Path(raw_path)
    if p.exists():
        return p
    # 2. Cari berdasarkan nama file di dalam DATASET_DIR (misal: dataset/test/Parasitized/xxx.png)
    if DATASET_DIR.exists():
        matches = list(DATASET_DIR.rglob(image_id))
        if matches:
            return matches[0]
    return None

def main():
    predictions = {}
    path_lookup = {}

    # 1. Muat YOLO26n-cls
    yolo_csv = BASE_DIR / "malaria_yolo26n_150ep_val_loss_lr5e4_test_predictions.csv"
    df_yolo = pd.read_csv(yolo_csv)
    df_yolo["image_id"] = df_yolo["image_path"].map(lambda p: Path(p).name)
    path_lookup.update(dict(zip(df_yolo["image_id"], df_yolo["image_path"])))
    predictions["YOLO26n-cls"] = pd.DataFrame({
        "image_id": df_yolo["image_id"],
        "y": (df_yolo["true_label"] == "Parasitized").astype(int),
        "pred": (df_yolo["pred_label"] == "Parasitized").astype(int),
        "score": df_yolo["prob_parasitized"],
    })

    # 2. Muat Custom CNN
    cnn_csv = BASE_DIR / "cnn_150ep" / "cnn_test_predictions.csv"
    df_cnn = pd.read_csv(cnn_csv)
    df_cnn["image_id"] = df_cnn["image_path"].map(lambda p: Path(p).name)
    path_lookup.update(dict(zip(df_cnn["image_id"], df_cnn["image_path"])))
    predictions["Custom CNN"] = df_cnn[["image_id", "y", "pred", "score"]].copy()

    # 3. Muat 4 Varian SVM (SVM Pixel, SVM Local, SVM Global, SVM Local + Global)
    svm_csv = BASE_DIR / "svm" / "svm_test_predictions.csv"
    df_svm = pd.read_csv(svm_csv)
    for model_name, group in df_svm.groupby("model", sort=False):
        predictions[model_name] = group[["image_id", "y", "pred", "score"]].reset_index(drop=True)

    order = [
        "YOLO26n-cls",
        "Custom CNN",
        "SVM Pixel",
        "SVM Local",
        "SVM Global",
        "SVM Local + Global",
    ]
    models = [m for m in order if m in predictions]

    # Gabungkan prediksi berdasarkan irisan image_id di keenam model
    common_ids = sorted(set.intersection(*[set(predictions[m]["image_id"]) for m in models]))
    print(f"Total sampel irisan ke-6 model: {len(common_ids)}")

    # Ambil 6 sampel (3 Parasitized, 3 Uninfected) — atau prioritaskan sampel dengan perbedaan prediksi antarmodel
    merged = predictions[models[0]][predictions[models[0]]["image_id"].isin(common_ids)][["image_id", "y"]].copy()
    for m in models:
        sub = predictions[m].set_index("image_id")
        merged[m] = merged["image_id"].map(sub["pred"])

    # Pilih 3 sampel Parasitized dan 3 sampel Uninfected
    sample_pos = merged[merged["y"] == 1].head(3)
    sample_neg = merged[merged["y"] == 0].head(3)
    samples = pd.concat([sample_pos, sample_neg]).reset_index(drop=True)

    # Buat grid visualisasi 2 baris x 3 kolom
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()

    for idx, row in samples.iterrows():
        img_id = row["image_id"]
        true_y = int(row["y"])
        ax = axes[idx]

        img_file = find_image_file(img_id, path_lookup.get(img_id, ""))
        if img_file and img_file.exists():
            ax.imshow(Image.open(img_file))
        else:
            ax.text(0.5, 0.5, f"Gambar tidak ditemukan:\n{img_id}", ha="center", va="center", fontsize=9)
            ax.set_facecolor("#f0f0f0")

        lines = [f"File: {img_id}", f"Label Asli: {label_str(true_y)}", "-" * 32]
        for m in models:
            pred_y = int(row[m])
            status = "✓" if pred_y == true_y else "✗"
            lines.append(f"{status} {m}: {label_str(pred_y)}")

        ax.set_title("\n".join(lines), fontsize=9.5, loc="left", family="monospace")
        ax.axis("off")

    plt.suptitle("Perbandingan Label Asli vs Prediksi Enam Model", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Visualisasi berhasil disimpan di: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()