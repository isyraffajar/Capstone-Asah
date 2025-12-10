import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


rfm = pd.read_csv("../Dataset/rfm.csv", encoding="latin1")
df = pd.read_csv("../Dataset/OnlineRetail.csv", encoding="latin1")

st.title("Capstone Project - Asah")
st.sidebar.title("Page")
st.sidebar.markdown("Select a page to navigate through the app.")
page = st.sidebar.selectbox("Choose your page", ["Home", "Data Exploration", "Modeling", "Insights"])

if page == "Home":
    st.header("Welcome to the Capstone Project - Asah")
    st.subheader("Dataset Overview")

    st.dataframe(df)

    # Tampilkan fitur/kolom
    st.subheader("Dataset Features")
    st.write(df.columns.tolist())

    # Tampilkan describe
    st.subheader("Dataset Information")
    summary = pd.DataFrame({
            "dtype": df.dtypes,
            "missing": df.isnull().sum(),
            "unique": df.nunique(),
            })
    st.dataframe(summary)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())

elif page == "Data Exploration":
    st.header("Data Exploration")
    st.write("RFM Analysis")

    # Chart Recency
    st.subheader("Distribusi Recency")
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.histplot(rfm['Recency'], bins=50, kde=True, ax=ax1)
    ax1.set_xlabel("Recency (hari)")
    ax1.set_ylabel("Jumlah Customer")
    st.pyplot(fig1)

    # Chart Frequency
    st.subheader("Distribusi Frequency")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.histplot(rfm['Frequency'], bins=40, kde=True, ax=ax2)
    ax2.set_xlabel("Jumlah Transaksi")
    ax2.set_ylabel("Jumlah Customer")
    st.pyplot(fig2)

    # Chart Monetary
    st.subheader("Distribusi Monetary")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.histplot(rfm['Monetary'], bins=50, kde=True, ax=ax3)
    ax3.set_xlabel("Total Belanja")
    ax3.set_ylabel("Jumlah Customer")
    st.pyplot(fig3)
    st.markdown("""
    ### Insight yang didapat

    *1. Distribusi Recency (Keterkinian)*
                
    Plot: Distribusi menurun tajam di awal(right-skewed).
    Insight:
    - Basis Pelanggan Aktif Tinggi: Sebagian besar pelanggan memiliki nilai Recency yang sangat rendah (sekitar 0 hingga 50 hari). Ini berarti sebagian besar transaksi terjadi baru-baru ini, menunjukkan basis pelanggan yang cukup aktif dan terlibat dalam periode waktu yang dekat.
    
    - Potensi Churn/Pelanggan Lama: Terdapat sejumlah kecil pelanggan dengan Recency tinggi (mendekati 350 hari). Pelanggan ini sudah lama tidak bertransaksi dan mungkin berisiko churn (berhenti membeli).

    *2. Distribusi Frequency (Frekuensi)*
                
    Plot: Distribusi sangat miring ke kanan (heavily right-skewed) dengan puncak yang sangat tinggi di sekitar nol.
    Insight:
    - Pelanggan Sekali Beli Dominan: Mayoritas pelanggan (sekitar 3500+ pelanggan) hanya melakukan 1 hingga 2 transaksi saja. Ini menunjukkan bahwa bisnis ini memiliki banyak pelanggan yang hanya mencoba atau membeli sekali, tetapi sulit untuk mempertahankan mereka agar membeli lagi (repeat purchase).

    - Pelanggan Loyal Kecil: Hanya segelintir pelanggan yang memiliki Frequency sangat tinggi (misalnya di atas 50 transaksi). Kelompok ini adalah pelanggan loyal yang sangat berharga dan harus dipertahankan.

    *3. Distribusi Monetary (Moneter)*
                
    Plot: Distribusi juga sangat miring ke kanan, dengan puncak yang sangat tinggi mendekati nol.
    Insight:
    - Nilai Belanja Kecil Dominan: Mayoritas pelanggan memiliki Total Belanja yang relatif kecil (berada di titik awal sumbu x). Hal ini sesuai dengan temuan Frequency, di mana banyak pelanggan hanya membeli sekali dengan jumlah kecil.

    - Pelanggan Bernilai Tinggi: Meskipun jumlahnya sedikit, terdapat beberapa pelanggan yang memiliki nilai Monetary sangat tinggi (hingga lebih dari Rp 2.500.000). Kelompok ini dikenal sebagai High-Value Customers atau Whales yang menyumbang persentase signifikan dari total pendapatan.
        
    """)
    df_corr = rfm.drop(columns=["CustomerID"])
    corr = df_corr.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Reds", ax=ax)
    ax.set_title("Heatmap Korelasi RFM")
    st.pyplot(fig)
    st.markdown("""
    ### Insight Korelasi RFM
    1. Korelasi Positif Kuat:
        - Frequency dan Monetary (0.55):

            Insight: Terdapat korelasi positif yang cukup kuat. Artinya, pelanggan yang lebih sering berbelanja (Frequency tinggi) cenderung memiliki total pengeluaran -yang lebih besar (Monetary tinggi).

            Implikasi: Frequency adalah prediktor yang baik untuk Monetary. Strategi untuk meningkatkan frekuensi pembelian akan secara langsung berkontribusi pada peningkatan pendapatan.

    2. Korelasi Negatif Sedang:
        - Recency dan Frequency (-0.26):

            Insight: Terdapat korelasi negatif yang sedang. Artinya, semakin baru pelanggan bertransaksi (Recency rendah), semakin sering mereka cenderung berbelanja (Frequency tinggi).

            Implikasi: Pelanggan yang aktif saat ini (Recency rendah) adalah juga pelanggan yang sering membeli. Penting untuk menjaga pelanggan aktif ini agar tidak menjadi "lama" (Recency tinggi).

        - Recency dan Monetary (-0.12):

            Insight: Terdapat korelasi negatif yang lemah. Artinya, pelanggan yang lebih baru bertransaksi (Recency rendah) cenderung memiliki Monetary yang sedikit lebih tinggi, namun hubungannya tidak sekuat dengan Frequency.

    Implikasi: Meskipun ada kecenderungan bahwa pelanggan baru mungkin berkontribusi lebih banyak pada pendapatan, faktor lain juga mempengaruhi Monetary. Strategi retensi pelanggan baru tetap penting, tetapi fokus utama harus pada peningkatan frekuensi pembelian.
    """)

    st.subheader("1. Recency vs Monetary")
    fig4, ax4 = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', ax=ax4)
    ax4.set_xlabel("Recency (hari)")
    ax4.set_ylabel("Monetary (Rp)")
    ax4.set_title("Scatter Plot Recency vs Monetary")
    st.pyplot(fig4)
    st.markdown("""
    ##### Insight Recency vs Monetary (Recency di Sumbu X, Monetary di Sumbu Y)
    Pola Umum: Sebagian besar titik data berkumpul di area Recency rendah (dekat 0 hari) dan Monetary rendah (dekat 0).

    Hubungan: Secara umum, ketika Recency meningkat (semakin lama pelanggan tidak membeli), Monetary cenderung turun, atau setidaknya tidak ada peningkatan Monetary yang jelas. Ini sesuai dengan korelasi negatif yang lemah (-0.12).

    Insight Penting (Outliers): Ada beberapa outlier di area Recency rendah (baru beli) yang memiliki Monetary sangat tinggi (di atas 100.000, bahkan hingga 250.000). Ini adalah Pelanggan Terbaik (Best Customers). Ada juga beberapa outlier dengan Recency sangat tinggi (sudah lama tidak beli, >300 hari) tetapi Monetary mereka tetap rendah.
    Implikasi Bisnis: Fokus pada pelanggan dengan Recency rendah karena mereka cenderung memberikan kontribusi lebih besar terhadap pendapatan. Strategi retensi untuk pelanggan yang sudah lama tidak bertransaksi juga penting, meskipun dampaknya terhadap Monetary mungkin tidak sebesar pelanggan aktif.
    """)

    st.subheader("2. Monetary vs Frequency")
    fig5, ax5 = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', ax=ax5)
    ax5.set_xlabel("Frequency (jumlah transaksi)")
    ax5.set_ylabel("Monetary (Rp)")
    ax5.set_title("Scatter Plot Monetary vs Frequency")
    st.pyplot(fig5)
    st.markdown("""
    ##### Insight Monetary vs Frequency (Monetary di Sumbu X, Frequency di Sumbu Y)
    Pola Umum: Sebagian besar titik data berkumpul di area Monetary dan Frequency yang rendah.

    Hubungan: Ada kecenderungan yang jelas, yaitu semakin tinggi Monetary (pengeluaran total), semakin tinggi pula Frequency (frekuensi pembelian). Pola ini mendukung korelasi positif yang kuat (0.55).

    Insight Penting (Outliers):Terdapat beberapa Pelanggan Berharga Tinggi yang:
    - Memiliki Monetary sangat tinggi (hingga 250.000) meskipun Frequency-nya moderat (di bawah 50 kali). Ini mungkin pelanggan yang melakukan pembelian besar, tetapi jarang.

    - Memiliki Frequency sangat tinggi (hingga 200 kali) meskipun Monetary-nya relatif moderat. Ini mungkin pelanggan yang melakukan pembelian kecil, tetapi sangat sering.
    """)

    st.subheader("3. Frequency vs Recency")
    fig6, ax6 = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=rfm, x='Recency', y='Frequency', ax=ax6)
    ax6.set_xlabel("Recency (hari)")
    ax6.set_ylabel("Frequency (jumlah transaksi)")
    ax6.set_title("Scatter Plot Frequency vs Recency")
    st.pyplot(fig6)
    st.markdown("""
    ##### Insight Frequency vs Recency (Frequency di Sumbu X, Recency di Sumbu Y)
    Pola Umum: Titik data sangat terkonsentrasi di sudut kiri bawah (Frequency rendah dan Recency rendah). Ini menunjukkan bahwa banyak pelanggan baru saja membeli, tetapi hanya sekali atau dua kali.

    Hubungan: Grafik menunjukkan hubungan non-linier yang kuat: Ketika Frequency meningkat, Recency cenderung menurun dengan cepat. Artinya, pelanggan yang sering membeli adalah pelanggan yang baru-baru ini membeli. Pola ini mendukung korelasi negatif sedang (-0.26).

    Insight Penting: Garis kepadatan data yang curam ini menunjukkan bahwa keterkinian pembelian (Recency) sangat bergantung pada seberapa sering pelanggan membeli (Frequency). Jika Anda berhenti membeli, Anda dengan cepat menjadi pelanggan "lama" (Recency tinggi).
    """)

elif page == "Modeling":
    st.header("Modeling")
    st.markdown("""
    ### Ringkasan Model yang Digunakan

    Pada notebook `CustomerSegmentation.ipynb` kami menggunakan dua pendekatan untuk menentukan dan membangun model clustering berbasis KMeans. Kedua versi tersebut sama-sama menggunakan KMeans namun berbeda pada metode pemilihan jumlah cluster (`k`):

    1. **Elbow Method (Inertia)**
        - Prinsip: Mengukur `inertia` (total sum of squared distances dari tiap titik ke pusat cluster). Untuk variasi `k`, kita plot inertia vs `k`.
        - Tujuan: Cari titik 'elbow' di grafik di mana penurunan inertia mulai melambat — itu seringkali menjadi pilihan `k` yang baik.
        - Kelebihan: Sederhana dan cepat; memberikan gambaran visual tentang trade-off antara jumlah cluster dan reduksi error.
        - Keterbatasan: Kadang elbow tidak jelas (kurva halus) sehingga pemilihan `k` bisa subjektif.
        *Berdasarkan metode ini kami mendapatkan jumlah cluster optimal `k = 3`.*

    2. **Silhouette Score**
        - Prinsip: Untuk tiap titik, silhouette mengukur seberapa mirip titik dengan cluster-nya sendiri dibanding cluster terdekat lainnya. Nilainya berada di rentang -1 sampai +1.
        - Tujuan: Untuk setiap `k`, hitung skor silhouette rata‑rata; pilih `k` dengan skor tertinggi.
        - Kelebihan: Memberikan ukuran kuantitatif kualitas cluster (kohesi vs pemisahan).
        - Keterbatasan: Bisa bias terhadap cluster berukuran seimbang dan kurang cocok bila ada banyak outlier/skala berbeda.
        *Berdasarkan metode ini kami mendapatkan jumlah cluster optimal `k = 5.*
    
    Berdasarkan kedua pendekatan tersebut, kami menyimpan dua versi model KMeans (dengan `k=3` dan `k=5`) untuk dianalisis dan dibandingkan lebih lanjut.
    ---

    **Pra‑proses yang dilakukan sebelum clustering**
    - Standarisasi/normalisasi (`StandardScaler`) pada fitur RFM (Recency, Frequency, Monetary) untuk menyamakan skala.
    - Menggunakan metrik `Recency` (hari sejak transaksi terakhir berdasarkan `snapshot_date`), `Frequency` (jumlah transaksi unik), dan `Monetary` (total belanja).

    **Bagaimana kami menggunakan kedua pendekatan**
    - Kami menjalankan KMeans pada data RFM yang telah discaling dan mengevaluasi beberapa nilai `k` menggunakan Elbow Method dan Silhouette Score.
    - Hasilnya kami simpan dua versi model (versi `k` dari Elbow dan versi `k` dari Silhouette) untuk dibandingkan lebih lanjut dengan visualisasi (PCA dan plot 2D/3D) dan silhouette score final.

    **Interpretasi hasil & saran bisnis**
    - Gunakan centroid cluster (rata‑rata Recency, Frequency, Monetary per cluster) untuk memberi label bisnis: mis. `Best Customers`, `Potential Loyal`, `At Risk`, `Low Value`.
    - Targeting: kirim promosi re‑engagement ke cluster dengan Recency tinggi; loyalty program untuk cluster dengan Frequency tinggi.
    - Validasi: selain metrik internal (inertia/silhouette), lakukan pengecekan manual terhadap ukuran cluster dan representasi pelanggan bisnis.

    Jika Anda mau, saya bisa tambahkan ringkasan centroid dan contoh label cluster otomatis pada halaman ini.
    """)

    
    

elif page == "Insights":
        st.header("Insights")
        st.markdown("""
        **Tentang Proyek - Ringkasan Analisis**

        - **Data:** Kami menggunakan dataset Online Retail UCI dari kaggle. Tahap pra‑proses meliputi: konversi `InvoiceDate` ke datetime, buang baris tanpa `CustomerID`, hapus duplikasi, filter `Quantity > 0` dan `UnitPrice > 0`, serta membuat kolom `TotalPrice`.

        - **Exploratory Data Analysis (EDA):** Analisis `Top Products`, `Top Customers`, distribusi transaksi per `Hour`, distribusi `UnitPrice`, dan jumlah transaksi per `Country` untuk memahami pola pembelian.

        - **Perhitungan RFM:** Menghitung metrik RFM untuk masing‑masing `CustomerID`:
            - `Recency`: selisih hari antara `snapshot_date` dan tanggal transaksi terakhir pelanggan (di notebook `snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)`).
            - `Frequency`: jumlah `InvoiceNo` unik per pelanggan.
            - `Monetary`: total belanja (`TotalPrice`) per pelanggan.
            Hasil RFM disimpan ke file `../Dataset/rfm.csv`.

        - **Visualisasi & Insight:** Plot distribusi RFM (histogram), heatmap korelasi, serta scatterplot untuk hubungan antar metrik (Recency vs Monetary, Monetary vs Frequency, Frequency vs Recency). Insight bisnis diekstrak (mis. identifikasi pelanggan bernilai tinggi, pelanggan berisiko churn).

        - **Clustering (Customer Segmentation):** Standarisasi fitur RFM (`StandardScaler`) lalu lakukan KMeans clustering. Penentuan jumlah cluster (`k`) dievaluasi menggunakan dua pendekatan: Elbow Method (inertia) dan Silhouette Score. Hasil divisualisasikan dengan PCA dan plot 2D/3D, serta dievaluasi dengan `silhouette_score`.

        - **Rekomendasi singkat:** Gunakan centroid cluster untuk memberi label bisnis (contoh: `Best Customers`, `Potential Loyal`, `At Risk`, `Low Value`), jalankan kampanye re‑engagement pada cluster dengan `Recency` tinggi, dan program loyalitas pada cluster dengan `Frequency` tinggi. Lakukan validasi bisnis sebelum menerapkan tindakan otomatis.

        """)