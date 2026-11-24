# Paket Lite - Tiga Model Terpilih

Paket ini berisi model YOLO26n-cls, Custom CNN, dan SVM Local + Global beserta checkpoint/artefak evaluasinya.

## Model Terpilih

1. YOLO26n-cls, 150 epoch, learning rate 5e-4, validation-loss patience
2. Custom CNN, 150 epoch, learning rate 5e-4
3. SVM Local + Global

## Hasil Test

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| YOLO26n-cls | 97.21% | 98.29% | 96.08% | 97.17% | 99.57% |
| Custom CNN | 95.50% | 97.07% | 93.83% | 95.42% | 98.59% |
| SVM Local + Global | 93.11% | 94.53% | 91.51% | 92.99% | 97.88% |

## Isi

- `10_train_yolo.py`, `20_train_cnn.py`, `30_train_svm.py`, `40_compare.py`, `00_prepare_split.py`, dan `common.py`: kode reproduksi.
- `yolo26n_cls_150ep_val_loss_lr5e4/weights/best.pt`: checkpoint YOLO terbaik.
- `cnn_150ep/best_custom_cnn.pth`: checkpoint CNN terbaik.
- `svm/svm_local_plus_global.joblib`: model SVM Local + Global terbaik.
- Metrik, prediksi test, confusion matrix, dan riwayat training untuk model terpilih.

Dataset citra tidak disertakan untuk mempertahankan ukuran paket tetap kecil.

## Hasil Pengukuran Kecepatan CPU

Pengukuran dilakukan pada **AMD Ryzen 5 5600**, menggunakan **12 thread CPU**, tanpa GPU, pada seluruh test set sebanyak **2.756 citra**.

| Model | Ukuran checkpoint | Waktu total | Kecepatan rata-rata |
|---|---:|---:|---:|
| YOLO26n-cls | 15 MB | 12,64 detik | 218 citra/detik atau 4,59 ms/citra |
| Custom CNN | 386 KB | 19,44 detik | 142 citra/detik atau 7,05 ms/citra |
| SVM Local + Global | 333 KB | 0,257 detik | 10.736 citra/detik atau 0,093 ms/citra |

Skrip benchmark memakai batch 1 dan 20 citra sebagai pemanasan. Untuk YOLO dan CNN, pengukuran mencakup pembacaan citra, preprocessing, dan inferensi model. Untuk SVM, pengukuran hanya mencakup prediksi dari delapan fitur gabungan; ekstraksi fitur ORB dan GLCM belum dicatat sebagai waktu terpisah.

Hasil detail tersedia pada `BENCHMARK.md`, `benchmark_results.csv`, dan `benchmark_models.py`.
