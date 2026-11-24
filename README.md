# Klasifikasi Citra Sel Darah Malaria

Proyek ini disusun untuk mata kuliah **Pengolahan Citra Digital dan Visi Komputer**, dengan fokus membandingkan enam pendekatan klasifikasi citra sel darah malaria:

1. **YOLO26n-cls** (150 epoch, learning rate `5e-4`)
2. **Custom CNN** (150 epoch, learning rate `5e-4`)
3. **SVM Pixel**
4. **SVM Local**
5. **SVM Global**
6. **SVM Local + Global**

## Anggota Kelompok

| Nama | NIM |
| :--- | :--- |
| Catherine Nathania | 235150201111042 |
| Shinta Oktavia Ramadhani | 235150207111036 |
| Rifqi Cahyono | 235150201111002 |

## Dataset

Dataset yang digunakan bersumber dari Kaggle: Cell Images for Detecting Malaria (https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

- **Total citra:** 27.558 citra (kelas seimbang).
- **Pembagian data (80/10/10):** 22.046 train / 2.756 validation / 2.756 test.
- **Random seed:** 42.

## Hasil Utama

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| YOLO 150 epoch val-loss | 97.21% | 98.29% | 96.08% | 97.17% | 99.57% |
| Custom CNN 150 epoch | 95.50% | 97.07% | 93.83% | 95.42% | 98.59% |
| SVM Pixel | 81.68% | 79.43% | 85.49% | 82.35% | 90.87% |
| SVM Local | 92.60% | 94.20% | 90.78% | 92.46% | 94.15% |
| SVM Global | 86.76% | 86.00% | 87.81% | 86.89% | 93.41% |
| SVM Local + Global | 93.11% | 94.53% | 91.51% | 92.99% | 97.88% |

Model terbaik pada eksperimen ini adalah **YOLO26n-cls** dengan accuracy **97.21%**, F1-score **97.17%**, dan ROC-AUC **99.57%**[cite: 2].

## Konfigurasi Eksperimen

- **Citra YOLO / Custom CNN:** ukuran `224 x 224`
- **SVM Pixel:** fitur piksel pada resolusi `32 x 32` RGB
- **SVM Local:** dua fitur ORB
- **SVM Global:** enam fitur GLCM dan shape
- **SVM Local + Global:** delapan fitur gabungan (ORB + GLCM & shape)[cite: 2]

## Struktur Proyek

- `00_prepare_split.py`: membuat split train/validation/test dengan rasio 80/10/10.
- `10_train_yolo.py`: melatih dan mengevaluasi model `YOLO26n-cls`.
- `20_train_cnn.py`: melatih dan mengevaluasi model `Custom CNN`.
- `30_train_svm.py`: melatih empat representasi SVM.
- `40_compare.py`: membandingkan hasil prediksi seluruh model.
- `malaria_cls_split/`: direktori dataset yang telah dibagi.
- `malaria_split_manifest.csv`: daftar citra dan pembagian split deterministik.
- `cnn_150ep/`: checkpoint, riwayat training, prediksi, dan metrik Custom CNN.
- `runs/yolo26n_cls_150ep_val_loss_lr5e4/`: hasil training dan checkpoint YOLO.
- `svm/`: checkpoint, grid validation, prediksi, dan metrik SVM.
- `laporan/`: tabel ringkas dan visualisasi perbandingan enam model[cite: 2].
