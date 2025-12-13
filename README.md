# Customer Segmentation Dashboard (Capstone Project - Asah) 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-KMeans-orange)

## 📌 Gambaran Proyek
Proyek ini bertujuan untuk melakukan segmentasi pelanggan pada data transaksi ritel online (`OnlineRetail.csv`) menggunakan metode **RFM (Recency, Frequency, Monetary)** dan algoritma **K-Means Clustering**.

Hasil analisis disajikan dalam bentuk Dashboard Interaktif berbasis **Streamlit**, yang memungkinkan pengguna bisnis untuk melihat statistik data, visualisasi cluster, dan mendapatkan *insights* mengenai kelompok pelanggan (seperti pelanggan terbaik, pelanggan berisiko, dll).

## 📂 Struktur Folder
Agar aplikasi berjalan lancar, pastikan struktur folder proyek Anda seperti berikut:

```text
├── Assets/
├── Dataset/
│   ├── OnlineRetail.csv            # Dataset mentah (sumber utama)
│   ├── rfm.csv                     # Output dari Notebook (Data RFM)
│   └── rfm_with_clusters.csv       # Output dari Notebook (Data + Label Cluster)
├── Code/
│   ├── CustomerSegmentation.ipynb  # Notebook untuk Data Cleaning, EDA, & Modeling
│   ├── ui.py                       # Source code Dashboard Streamlit
├── requirements.txt                # Daftar library python
└── README.md                       # Dokumentasi ini
```
## 📥 Download Model

Model machine learning (`model_clustering.pkl`) yang sudah dilatih disimpan secara terpisah di Google Drive. Silakan unduh dan letakkan di folder proyek Anda sebelum menjalankan aplikasi.

🔗 **Link Google Drive:**

> **[https://drive.google.com/file/d/1XXNZrmJGnPSLFf3FtI7LUFGPlTtNvICh/view?usp=sharing]**

*Catatan: Jika Anda ingin melatih model sendiri dari awal, Anda bisa melewati langkah ini dan menjalankan `CustomerSegmentation.ipynb`.*
## 🛠️ Teknologi & Library

  * **Python**: Bahasa pemrograman utama.
  * **Streamlit**: Framework untuk membuat antarmuka web dashboard.
  * **Pandas & NumPy**: Manipulasi dan analisis data tabular.
  * **Scikit-Learn**:
      * `StandardScaler`: Normalisasi data.
      * `KMeans`: Algoritma clustering.
      * `silhouette_score`: Evaluasi kualitas cluster.
  * **Matplotlib & Seaborn**: Visualisasi grafik statis.

## 🚀 Fitur Dashboard (Streamlit)

Aplikasi `ui.py` memiliki menu navigasi di sidebar dengan fitur sebagai berikut:

### 1\. 🏠 Home

  * Menampilkan gambaran umum dataset.
  * Statistik deskriptif data.
  * Informasi tipe data dan *missing values*.

### 2\. 🔍 Data Exploration

  * Visualisasi distribusi data menggunakan Boxplot (Quantity & Unit Price).
  * Analisis persebaran transaksi berdasarkan negara (Pie Chart).

### 3\. 🤖 Modeling (Clustering)

  * **Evaluasi Model**: Grafik *Elbow Method* dan *Silhouette Score* untuk menentukan jumlah cluster optimal ($K$).
  * **Visualisasi Cluster**: Scatter plot interaktif yang membandingkan:
      * Recency vs Frequency
      * Frequency vs Monetary
      * Recency vs Monetary
  * **Distribusi Cluster**: Boxplot untuk melihat karakteristik R, F, dan M pada setiap cluster.

### 4\. 💡 Insights

Halaman ini memberikan rekomendasi bisnis dengan memfilter pelanggan ke dalam kategori:

  * **Cluster Summary**: Rata-rata metrik RFM per cluster.
  * **Most Spent Customers**: Pelanggan dengan pengeluaran (`Monetary`) tertinggi di cluster tertentu.
  * **Best Customers**: Pelanggan dengan skor RFM terbobot tertinggi.
  * **Newest Customers**: Pelanggan yang baru saja bertransaksi namun frekuensinya belum tinggi.
  * **At Risk Customers**: Pelanggan yang dulu sering bertransaksi tapi sudah lama tidak kembali (Recency tinggi).
  * **Regular Customers**: Pelanggan dengan pola transaksi rata-rata (menengah).

## ⚙️ Cara Menjalankan

### Langkah 1: Instalasi

Pastikan Python sudah terinstall, lalu install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### 2\. Persiapan Model & Data

Anda memiliki dua opsi:

  * **Opsi A (Cepat):** Download `model_clustering.pkl` dari link Google Drive di atas, lalu pastikan file CSV (`rfm.csv`, `rfm_with_clusters.csv`) sudah tersedia di folder `Dataset/`.
  * **Opsi B (Training Ulang):** Buka dan jalankan semua sel di `CustomerSegmentation.ipynb` untuk menghasilkan file model dan data CSV baru.

### Langkah 3: Menjalankan Dashboard

Ada dua opsi untuk menjalankan dashboard, yaitu lewat localhost atau versi deploy. Berikut cara menjalankan keduanya :
#### 1. Jalankan di Browser yang sudah di deploy

Untuk run web yang sudah di deploy, tinggal buka link ini : [asah-a25-cs305.streamlit.app](https://asah-a25-cs305.streamlit.app)


#### 2. jalankan di Browser localhost

Buka folder Projeknya.

Buka new Terminal (Ctrl + Shift + ') jika di VSCode.

Jalankan perintah berikut di terminal:

```bash
streamlit run Code/ui.py
```

Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`). (Bisa berbeda tiap device).

-----

**Created for Capstone Project - Asah**
