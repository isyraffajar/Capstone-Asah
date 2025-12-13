# Customer Segmentation System (Capstone Project - Asah) 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-KMeans-orange)

## 📌 Gambaran Proyek
Proyek ini adalah sistem end-to-end untuk segmentasi pelanggan ritel online. Proyek ini terdiri dari dua aplikasi utama:
1.  **Dashboard Analisis (`ui.py`)**: Untuk menganalisis data historis, memvisualisasikan cluster, dan melihat wawasan bisnis.
2.  **Aplikasi Prediksi (`app.py`)**: Untuk memprediksi segmen pelanggan baru, baik secara satuan (input manual) maupun massal (upload CSV).

Metode yang digunakan adalah **RFM (Recency, Frequency, Monetary)** dan algoritma **K-Means Clustering**.

## 📂 Struktur Folder
Agar kedua aplikasi berjalan lancar, pastikan struktur folder proyek Anda seperti berikut:

```text
├── Dataset/
│   ├── OnlineRetail.csv        # Dataset mentah
│   ├── rfm.csv                 # Data hasil RFM
│   └── rfm_with_clusters.csv   # Data final dengan cluster
├── model/ (atau Code/)
│   ├── model_clustering.pkl    # Model KMeans yang sudah dilatih
│   └── scaler.pkl              # Scaler untuk normalisasi data
├── CustomerSegmentation.ipynb  # Notebook pelatihan model
├── ui.py                       # Dashboard Analisis
├── app.py                      # Aplikasi Prediksi (New Features)
├── requirements.txt            # Daftar library python
└── README.md                   # Dokumentasi ini
````

## 📥 Download Model & Scaler (Wajib untuk app.py)

Aplikasi prediksi (`app.py`) membutuhkan dua file biner agar bisa berjalan. Karena ukuran file atau alasan portabilitas, file ini disimpan di Google Drive.

1.  **model\_clustering.pkl**: Model algoritma K-Means.
2.  **scaler.pkl**: Standard Scaler untuk menormalisasi input user agar sesuai dengan format model.

🔗 **Link Google Drive:**

> **[https://drive.google.com/file/d/1XXNZrmJGnPSLFf3FtI7LUFGPlTtNvICh/view?usp=sharing]**

*Instruksi: Unduh kedua file tersebut dan letakkan di dalam folder `model/` (atau sesuaikan dengan path di dalam `app.py` Anda).*

## 🛠️ Instalasi

Install library yang dibutuhkan (termasuk `plotly` untuk grafik di app baru):

```bash
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan Aplikasi

Anda memiliki dua pilihan aplikasi yang bisa dijalankan sesuai kebutuhan:

### 1\. Menjalankan Dashboard Analisis (Untuk Bisnis & Insight)

Gunakan aplikasi ini jika Anda ingin melihat performa data historis dan karakteristik tiap segmen.

```bash
streamlit run ui.py
```

**Fitur:**

  * Overview & Statistik Data.
  * Visualisasi Cluster (Scatter Plot & Box Plot).
  * Rekomendasi Bisnis (Best Customers, At Risk, dll).

### 2\. Menjalankan Aplikasi Prediksi (Untuk Operasional)

Gunakan aplikasi ini untuk menentukan segmen pelanggan baru secara *real-time*.

```bash
streamlit run app.py
```

**Fitur:**

  * **Input Manual (Sidebar)**: Masukkan nilai *Recency*, *Frequency*, dan *Monetary* satu per satu untuk melihat hasil segmen pelanggan tersebut secara instan.
  * **Batch Prediction**: Upload file CSV berisi data banyak pelanggan sekaligus.
      * *Syarat CSV*: Harus memiliki kolom `Recency`, `Frequency`, dan `Monetary`.
  * **Visualisasi Distribusi**: Grafik batang interaktif (Plotly) yang menunjukkan persebaran hasil prediksi.
  * **Download Hasil**: Unduh hasil segmentasi massal ke dalam format CSV.

## ⚙️ Pelatihan Ulang Model (Opsional)

Jika Anda ingin memperbarui model dengan data terbaru:

1.  Buka `CustomerSegmentation.ipynb`.
2.  Jalankan semua sel (*Run All*).
3.  Pastikan kode bagian bawah notebook menyimpan file `model_clustering.pkl` dan `scaler.pkl`.
4.  Pindahkan file output tersebut ke folder yang sesuai agar terbaca oleh `app.py`.

-----

**Created for Capstone Project - Asah**