# Laporan Enam Model Klasifikasi Malaria

## Dataset

Dataset berisi 27.558 citra yang seimbang:

- Parasitized: 13.779 citra.
- Uninfected: 13.779 citra.

Pembagian deterministik dengan seed 42:

- Train: 22.046 citra.
- Validation: 2.756 citra.
- Test: 2.756 citra.

## Model

Eksperimen membandingkan enam model utama:

1. YOLO26n-cls, learning rate 5e-4, maksimum 150 epoch, dengan validation-loss patience.
2. Custom CNN, learning rate 5e-4, maksimum 150 epoch.
3. SVM Pixel.
4. SVM Local menggunakan dua fitur ORB.
5. SVM Global menggunakan enam fitur GLCM dan shape.
6. SVM Local + Global menggunakan delapan fitur gabungan.

## Hasil Test

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| YOLO 150 ep val-loss | 97.21% | 98.29% | 96.08% | 97.17% | 99.57% |
| Custom CNN 150 Epoch LR 5e-4 | 95.50% | 97.07% | 93.83% | 95.42% | 98.59% |
| SVM Pixel | 81.68% | 79.43% | 85.49% | 82.35% | 90.87% |
| SVM Local | 92.60% | 94.20% | 90.78% | 92.46% | 94.15% |
| SVM Global | 86.76% | 86.00% | 87.81% | 86.89% | 93.41% |
| SVM Local + Global | 93.11% | 94.53% | 91.51% | 92.99% | 97.88% |

## Kesimpulan

YOLO26n-cls adalah model terbaik pada hasil test. Model ini memperoleh accuracy 97.21%, F1-score 97.17%, dan ROC-AUC 99.57%. Custom CNN berada di posisi kedua. Di antara konfigurasi SVM, Local + Global memberikan performa terbaik.
