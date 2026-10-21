# Proyek 6 — Data Analytics & Predictive Modeling

## Deskripsi Proyek

Proyek ini menerapkan analisis data dan machine learning untuk mempelajari tren penjualan bulanan serta membuat estimasi penjualan pada periode berikutnya.

Program menggunakan Python untuk membersihkan data, menghitung statistik deskriptif, mengukur pertumbuhan penjualan, melatih model regresi linear, dan mengevaluasi hasil prediksi.

## Tujuan

- Memahami proses analisis data menggunakan Python.
- Mengidentifikasi tren penjualan bulanan.
- Menghitung statistik dan pertumbuhan penjualan.
- Menerapkan regresi linear untuk memprediksi nilai numerik.
- Mengevaluasi model menggunakan MAE dan RMSE.
- Menyajikan hasil melalui grafik dan laporan CSV.

## Teknologi yang Digunakan

- Python
- Pandas
- Matplotlib
- Scikit-learn
- CSV

## Struktur Folder

```text
Proyek_6_Data_Analytics_Predictive_Modeling/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── penjualan_bulanan.csv
└── output/
    ├── analisis_penjualan.png
    ├── data_analisis.csv
    ├── evaluasi_model.csv
    └── prediksi_bulan_berikutnya.csv
```

Dataset contoh dibuat otomatis apabila file sumber belum tersedia.

## Cara Menjalankan

Instal dependensi:

```bash
python -m pip install -r requirements.txt
```

Jalankan program:

```bash
python main.py
```

## Fitur

1. Membuat dataset penjualan bulanan contoh.
2. Memvalidasi dan membersihkan data.
3. Menghitung total, rata-rata, nilai tertinggi, dan nilai terendah.
4. Menghitung persentase pertumbuhan bulanan.
5. Melatih model regresi linear.
6. Menguji model menggunakan periode data terakhir.
7. Mengukur kesalahan prediksi menggunakan MAE dan RMSE.
8. Mengestimasi penjualan untuk bulan berikutnya.
9. Membuat grafik tren dan pertumbuhan penjualan.
10. Mengekspor hasil analisis dan prediksi ke CSV.

## Metode Machine Learning

Model yang digunakan adalah **Linear Regression**. Model mempelajari hubungan antara urutan bulan dan nilai penjualan, lalu menggunakan pola tersebut untuk membuat estimasi bulan berikutnya.

Data periode terakhir digunakan sebagai data uji, sedangkan periode sebelumnya digunakan untuk pelatihan. MAE dan RMSE digunakan untuk mengukur selisih prediksi terhadap nilai aktual.

## Hasil Program

- `analisis_penjualan.png`: grafik tren, estimasi penjualan, dan pertumbuhan bulanan.
- `data_analisis.csv`: data beserta hasil perhitungan pertumbuhan.
- `evaluasi_model.csv`: hasil prediksi pada data uji dan selisih absolut.
- `prediksi_bulan_berikutnya.csv`: estimasi penjualan periode berikutnya beserta metrik evaluasi.

## Kesimpulan

Proyek ini menunjukkan proses analisis data dan penerapan regresi linear untuk memperkirakan penjualan berdasarkan tren historis. Hasil prediksi dapat membantu demonstrasi perencanaan berbasis data, tetapi tidak menjamin penjualan sebenarnya.

**Catatan:** Dataset yang digunakan merupakan data simulasi untuk pembelajaran. Model hanya menggunakan tren waktu sederhana dan belum mempertimbangkan faktor lain seperti harga, promosi, musim, atau kondisi pasar.
