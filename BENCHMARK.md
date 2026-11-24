# Hasil Pengukuran Kecepatan CPU

Pengukuran dilakukan pada **AMD Ryzen 5 5600**, menggunakan CPU **12 thread**, tanpa GPU, pada seluruh test set sebanyak 2.756 citra.

| Model | Ukuran checkpoint | Waktu total | Kecepatan rata-rata |
|---|---:|---:|---:|
| YOLO26n-cls | 15 MB | 12,64 detik | 218 citra/detik atau 4,59 ms/citra |
| Custom CNN | 386 KB | 19,44 detik | 142 citra/detik atau 7,05 ms/citra |
| SVM Local + Global | 333 KB | 0,257 detik | 10.736 citra/detik atau 0,093 ms/citra |

Waktu SVM hanya mengukur tahap prediksi 8 fitur gabungan. Waktu ekstraksi fitur ORB dan GLCM belum dilaporkan secara terpisah.
