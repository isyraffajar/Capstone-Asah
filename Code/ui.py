import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


rfm = pd.read_csv("Dataset/rfm.csv", encoding="latin1")
rfmc = pd.read_csv("Dataset/rfm_with_clusters.csv", encoding="latin1")
df = pd.read_csv("Dataset/OnlineRetail.csv", encoding="latin1")

st.title("Capstone Project - Asah")
st.sidebar.title("Page")
st.sidebar.markdown("Select a page to navigate through the app.")
page = st.sidebar.selectbox("Choose your page", ["Home", "Data Exploration", "Modeling", "Insights"])

if page == "Home":
    st.header("Welcome to the Capstone Project - Asah")
    st.subheader("Dataset Preview")

    PAGE_SIZE = 1000
    TOTAL_ROWS = len(df)
    TOTAL_PAGES = (TOTAL_ROWS - 1) // PAGE_SIZE + 1

    # Init state
    if "page" not in st.session_state:
        st.session_state.page = 1

    start = (st.session_state.page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, TOTAL_ROWS)

    # =====================
    # TABLE (ATAS)
    # =====================
    st.dataframe(df.iloc[start:end])

    st.caption(
        f"Showing rows {start + 1:,} – {end:,} of {TOTAL_ROWS:,}"
    )

    # =====================
    # PAGINATION (BAWAH)
    # =====================
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅ Prev", disabled=st.session_state.page == 1):
            st.session_state.page -= 1
            st.rerun()

    with col2:
        left, center, right = st.columns([1, 2, 1])

        with center:
            st.markdown(
                f"<p style='text-align:center; font-weight:600;'>"
                f"Page {st.session_state.page} / {TOTAL_PAGES}"
                f"</p>",
                unsafe_allow_html=True
            )

            page = st.number_input(
                "",
                min_value=1,
                max_value=TOTAL_PAGES,
                value=st.session_state.page,
                step=1,
                label_visibility="collapsed"
            )

            if page != st.session_state.page:
                st.session_state.page = page
                st.rerun()

    with col3:
        if st.button("Next ➡", disabled=st.session_state.page == TOTAL_PAGES):
            st.session_state.page += 1
            st.rerun()

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
    st.header("Explanatory Data Analysis (ExDA)")
    st.markdown("""

    Explanatory Data Analysis (ExDA) 
    adalah tahap analisis data yang berfokus pada penjelasan hasil temuan setelah proses eksplorasi dilakukan. 
    ExDA digunakan untuk menjelaskan mengapa pola tersebut terjadi, serta menghubungkan temuan analisis dengan konteks bisnis atau penelitian. 
    ExDA biasanya mencakup penyajian visualisasi yang terarah, interpretasi hubungan antar variabel, penjelasan perbedaan antar kelompok data, serta pemberian insight yang dapat digunakan untuk pengambilan keputusan.
        
    """)

    st.subheader("Grafik Top 10 Customer Spending")
    st.image("Assets/top10spending.png", use_container_width=True)
    st.markdown("""
    #### Insight yang didapat
                
    *1. Dominasi Spending oleh Customer Teratas*

    Terlihat jelas adanya konsentrasi spending yang signifikan pada beberapa customer teratas.
    - Customer ID 14646 memiliki total spending tertinggi, yaitu 279.138.
    - Customer ID 18102 berada di posisi kedua dengan spending 259.657.
    - Customer ID 17450 menempati posisi ketiga dengan spending 194.391.

    *2. Perbandingan Spending*
    
    Selisih antara customer teratas dengan customer di posisi berikutnya cukup besar, menunjukkan bahwa ada sekelompok kecil customer yang sangat loyal atau melakukan pembelian dalam jumlah besar (mereka adalah "Super Buyers" atau High-Value Customers).

    - Spending tertinggi (279.138) hampir 4 kali lipat dari spending terendah di Top 10 (72.708 oleh Customer ID 16029).
        
    """)

    st.subheader("Grafik Top 10 Prouk Unit Terjual Terbanyak")
    st.image("Assets/top10itemQuantity.png", use_container_width=True)
    st.markdown("""
    #### Insight yang didapat

    *1. Dominasi Item Teratas*
                
    Sama seperti spending customer, terjadi konsentrasi penjualan unit pada beberapa item teratas, yang menunjukkan item-item ini adalah "bintang" dalam hal volume transaksi.

    - Item No. 1: PAPER CRAFT, LITTLE BIRDIE adalah yang paling dominan dengan unit terjual mencapai 80.955 unit.

    - Item No. 2: MEDIUM CERAMIC TOP STORAGE JAR mengikuti dengan penjualan 77.916 unit.

    *2. Kesenjangan di Puncak*
    
    Selisih antara item teratas (80.955) dengan item di posisi ke-10 (MINI PAINT SET VINTAGE, 26.076) sangat besar. Volume item teratas lebih dari 3 kali lipat dari item di posisi ke-10.*

    *3. Penurunan Signifikan*

    Ada penurunan yang cukup besar dari Peringkat 2 (77.916) ke Peringkat 3 (WORLD WAR 2 GLIDERS ASSTD DESIGNS, 54.319). Penurunan ini sebesar 23.597 unit, yang merupakan salah satu penurunan volume terbesar di awal daftar.

    Penurunan volume menjadi lebih bertahap setelah Peringkat 5 (JUMBO BAG RED RETROSPOT, 46.078) hingga Peringkat 9.
        
    """)

    st.subheader("Grafik Top 10 Produk Paling Laku")
    st.image("Assets/top10itemSold.png", use_container_width=True)
    st.markdown("""
    #### Insight yang didapat

    *1. Item Paling Populer*

    - Item WHITE HANGING HEART T-LIGHT HOLDER adalah item yang paling sering dibeli, muncul dalam 2.023 transaksi. Ini menunjukkan item ini memiliki penerimaan pasar yang sangat tinggi dan sering menjadi pilihan customer.
    - Item REGENCY CAKESTAND 3 TIER berada di posisi kedua, muncul dalam 1.713 transaksi.

    *2. Distribusi Popularitas*
                
    - Terdapat sedikit penurunan dari peringkat 1 (2.023) ke peringkat 2 (1.713), sebesar 310 transaksi.
    - Item dari Peringkat 3 (JUMBO BAG RED RETROSPOT, 1.615 transaksi) hingga Peringkat 5 (PARTY BUNTING, 1.389 transaksi) menunjukkan popularitas yang relatif merata, dengan selisih sekitar 100-200 transaksi di antara mereka.
    - Peringkat 6 hingga Peringkat 10 memiliki transaksi di atas 1.000, menunjukkan bahwa semua item dalam daftar ini adalah item yang sangat sering dibeli dan memiliki basis pelanggan yang kuat.

    *3. Kategori Produk yang Populer*
    - Item-item yang sangat populer cenderung masuk dalam kategori Dekorasi (T-LIGHT HOLDER, BIRD ORNAMENT, PARTY BUNTING) dan Produk Penyimpanan/Dapur (REGENCY CAKESTAND 3 TIER, JUMBO BAG, LUNCH BAG).
    - Tiga dari sepuluh item teratas adalah Lunch Bag dengan desain berbeda (RED RETROSPOT, BLACK SKULL, SUKI DESIGN). Kehadiran tiga varian produk yang sama di Top 10 menegaskan bahwa Lunch Bag sebagai kategori produk secara keseluruhan memiliki permintaan yang sangat tinggi.
        
    """)

    st.subheader("Grafik Total Penjualan Dari Tiap Negara")
    st.image("Assets/pareto.png", use_container_width=True)
    st.markdown("""
    #### Insight yang didapat

    *1. Dominasi Mutlak oleh United Kingdom (UK)*
    
    - UK mendominasi total penjualan secara mutlak dengan kontribusi mencapai 95% dari total penjualan yang ditampilkan di grafik.
    - Nilai penjualan UK berada di sekitar $7.000.000, yang secara visual jauh lebih tinggi dari semua negara lain yang digabungkan.

    *2. Kontribusi Negara Lain Sangat Kecil*
    - Semua negara lain di luar UK hanya menyumbang 5% dari total penjualan.
    - Negara dengan kontribusi tertinggi setelah UK adalah Netherlands dengan 14% kontribusi kumulatif, atau 4% dari total (14% - 11% dari line chart kumulatif, sekitar $ 200.000 - $ 300.000).
    - EIRE (Irlandia), Germany, dan France juga memiliki kontribusi yang cukup signifikan di antara pasar non-UK, masing-masing menyumbang 11%, 8%, dan 6% kumulatif.

    *3. Konsentrasi Penjualan*
                
    karena bisnis ini berbasis di UK, maka dapat disimpulkan bahwa sebagian besar transaksinya berasal dari pasar domestik, dan hanya mencakup sebagian kecil dari pasar internasional. Meskipun demikian 
    ada potensi untuk ekspansi lebih lanjut di pasar internasional, terutama di negara-negara yang sudah menunjukkan kontribusi penjualan yang lebih tinggi seperti Netherlands, EIRE, dan Germany.
        
    """)

    
    st.header("RFM Analysis")
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

    KMeans adalah algoritma clustering yang membagi data ke dalam beberapa kelompok berdasarkan kemiripan. Algoritma ini bekerja dengan:

    1. Menentukan pusat awal untuk setiap cluster.
    2. Mengelompokkan data ke pusat terdekat.
    3. Memperbarui posisi pusat berdasarkan rata-rata data dalam cluster.
    4. Mengulang proses sampai pusat cluster stabil.

    KMeans membantu melihat pola pelanggan, segmentasi perilaku, atau kelompok nilai RFM secara jelas sehingga lebih mudah membuat strategi bisnis.
    """)
    
    st.markdown("""
    ---
    
    ### Pra-proses sebelum Clustering
    - Standarisasi/normalisasi (`StandardScaler`) pada fitur RFM (Recency, Frequency, Monetary) untuk menyamakan skala.
    - Men   gunakan metrik `Recency` (hari sejak transaksi terakhir berdasarkan `snapshot_date`), `Frequency` (jumlah transaksi unik), dan `Monetary` (total belanja).

    Untuk menentukan jumlah cluster (`k`), kami menggunakan:
    1. **Elbow Method (Inertia)**
        - Prinsip: Mengukur `inertia` (total sum of squared distances dari tiap titik ke pusat cluster). Untuk variasi `k`, kita plot inertia vs `k`.
        - Tujuan: Cari titik 'elbow' di grafik di mana penurunan inertia mulai melambat — itu seringkali menjadi pilihan `k` yang baik.
        - Kelebihan: Sederhana dan cepat; memberikan gambaran visual tentang trade-off antara jumlah cluster dan reduksi error.
    Untuk hasil kami, elbow terlihat pada `k=3`. Perhatikan chart berikut.
    """)

    st.image("Assets/elbowMethod.png", use_container_width=True)

    st.markdown("""
    Keterbatasan: Kadang elbow tidak jelas (kurva halus) sehingga pemilihan `k` bisa subjektif.

    2. **Silhouette Score**
        - Prinsip: Untuk tiap titik, silhouette mengukur seberapa mirip titik dengan cluster-nya sendiri dibanding cluster terdekat lainnya. Nilainya berada di rentang -1 sampai +1.
        - Tujuan: Untuk setiap `k`, hitung skor silhouette rata‑rata; pilih `k` dengan skor tertinggi.
        - Kelebihan: Memberikan ukuran kuantitatif kualitas cluster (kohesi vs pemisahan).
        - Keterbatasan: Bisa bias terhadap cluster berukuran seimbang dan kurang cocok bila ada banyak outlier/skala berbeda.
    Untuk hasil kami, silhouette score tertinggi terjadi pada `k=4`. Perhatikan chart berikut.
    """)

    st.image("Assets/silhouetteMethod.png", use_container_width=True)

    st.markdown("""
    
        Keterbatasan: Kadang skor bisa menyesatkan jika cluster sangat tidak seimbang atau data memiliki noise tinggi.        
    
    3. **Gabungan Elbow & Silhouette**
    
    Metode ini memakai pendekatan gabungan antara Elbow dan Silhouette. Nilai k awal ditentukan dari titik siku pada grafik Elbow, lalu k tersebut dievaluasi bersama k di sekitarnya (k–1 dan k+1). Setiap kandidat dihitung skor Silhouette-nya, kemudian k dengan skor tertinggi dipilih sebagai jumlah cluster final. Pendekatan ini membantu memilih jumlah cluster yang stabil dan memiliki pemisahan antar cluster yang baik.
    ```
    Jumlah cluster berdasarkan Elbow: 3
    Silhouette Scores untuk k>=3: > {3: 0.5940, 4: 0.6163, 5: 0.6041, 6: 0.5960, 7: 0.5186, 8: 0.4791, 9: 0.4261, 10: 0.4178}
    Jumlah cluster terpilih (gabungan Elbow & Silhouette): 4
    ```         
    ---
    
    ### Modeling dan Visualisasi Hasil
    - Kami menjalankan KMeans pada data RFM yang telah discaling dan mengevaluasi beberapa nilai `k` menggunakan Elbow Method dan Silhouette Score.
    - Hasilnya kami simpan dua versi model (versi `k` dari Elbow dan versi `k` dari Silhouette) untuk dibandingkan lebih lanjut dengan visualisasi (PCA dan plot 2D/3D) dan silhouette score final.

    #### 1. Scatter Plot Berdasarkan Pasangan Fitur
    Visualisasi ini membantu melihat persebaran cluster dalam ruang 2D:

    - Recency vs Monetary
    - Recency vs Frequency
    - Monetary vs Frequency

    """)

    st.image("Assets/RvsMclusterplot.png", caption="Cluster Plot (Recency vs Monetary)", use_container_width=True)
    st.image("Assets/RvsFclusterplot.png", caption="Cluster Plot (Recency vs Frequency)", use_container_width=True)
    st.image("Assets/FvsMclusterplot.png", caption="Cluster Plot (Monetary vs Frequency)", use_container_width=True)

    st.markdown("""

    #### 2. Visualisasi 3D RFM

    Visualisasi tiga dimensi memberikan gambaran menyeluruh tentang pembentukan cluster di ruang RFM.

    """)

    st.image("Assets/3DCluster.png", caption="3D Cluster Visualization (RFM)", use_container_width=True)

    st.markdown("""

    #### 3. PCA 2D Projection

    Dimensi data direduksi menggunakan PCA menjadi dua komponen utama untuk memudahkan visualisasi pola cluster. Proyeksi ini membantu melihat pemisahan border antar cluster tanpa kehilangan informasi penting.

    """)

    st.image("Assets/PCAVisualization.png", caption="Visualisasi Clustering KMeans (PCA)", use_container_width=True)

    st.markdown("""
    ---

    ### Ringkasan Halaman Modeling

    - Jumlah cluster optimal: `4`
    - Teknik evaluasi: **Elbow + Silhouette**
    - Skor evaluasi model: `0.6162696093073907`
    - Visualisasi menunjukkan pemisahan cluster yang jelas
    - Cluster dapat digunakan untuk segmentasi pelanggan berbasis RFM

    ---
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
            Hasil RFM disimpan ke file `Dataset/rfm.csv`.

        - **Visualisasi & Insight:** Plot distribusi RFM (histogram), heatmap korelasi, serta scatterplot untuk hubungan antar metrik (Recency vs Monetary, Monetary vs Frequency, Frequency vs Recency). Insight bisnis diekstrak (mis. identifikasi pelanggan bernilai tinggi, pelanggan berisiko churn).

        - **Clustering (Customer Segmentation):** Standarisasi fitur RFM (`StandardScaler`) lalu lakukan KMeans clustering. Penentuan jumlah cluster (`k`) dievaluasi menggunakan dua pendekatan: Elbow Method (inertia) dan Silhouette Score. Hasil divisualisasikan dengan PCA dan plot 2D/3D, serta dievaluasi dengan `silhouette_score`.

        - **Rekomendasi singkat:** Gunakan centroid cluster untuk memberi label bisnis (contoh: `Best Customers`, `Potential Loyal`, `At Risk`, `Low Value`), jalankan kampanye re‑engagement pada cluster dengan `Recency` tinggi, dan program loyalitas pada cluster dengan `Frequency` tinggi. Lakukan validasi bisnis sebelum menerapkan tindakan otomatis.

        """)
        st.dataframe(rfmc.groupby('Cluster')[['Recency','Frequency','Monetary']].mean())

        st.subheader("Ranking Cluster")
        # Hitung mean RFM tiap cluster
        cluster_stats = rfmc.groupby("Cluster").agg({
            "R_norm": "mean",
            "F_norm": "mean",
            "M_norm": "mean"
        }).reset_index()

        # Buat ranking
        cluster_stats["Recency_rank"] = cluster_stats["R_norm"].rank(ascending=True)   # Recency kecil lebih baik
        cluster_stats["Frequency_rank"] = cluster_stats["F_norm"].rank(ascending=False) # Frequency besar lebih baik
        cluster_stats["Monetary_rank"] = cluster_stats["M_norm"].rank(ascending=False)   # Monetary besar lebih baik

        # Total ranking
        cluster_stats["RFM_rank"] = (cluster_stats["Recency_rank"] + cluster_stats["Frequency_rank"] + cluster_stats["Monetary_rank"]) / 3

        # Urutkan cluster dari ranking terbaik ke terburuk
        cluster_stats = cluster_stats.sort_values("RFM_rank")

        # Print tabel
        print("Cluster dengan ranking terbaik berdasarkan RFM:")
        st.dataframe(cluster_stats)

        grouped_cluster = rfmc.groupby("Cluster")
        # 1. Tentukan kolom yang tidak ingin ditampilkan
        excluded_cols = ["CustomerID", "Cluster", "R_norm", "F_norm", "M_norm"]

        # 2. Buat list kolom target (ini akan menjadi Judul Tab)
        target_columns = [col for col in rfmc.columns if col not in excluded_cols]

        # 3. Buat Tabs berdasarkan list nama kolom tersebut
        tabs = st.tabs(target_columns)

        # 4. Loop secara bersamaan antara objek Tab dan Nama Kolom
        for tab, col_name in zip(tabs, target_columns):
            with tab:
                st.subheader(f"Statistik untuk {col_name}")
                
                # Menampilkan tabel describe
                st.dataframe(
                    grouped_cluster[col_name].describe(), 
                    use_container_width=True  # Agar tabel memenuhi lebar tab
                )

        st.subheader("Karakteristik tiap cluster")

        tab1, tab2, tab3, tab4 = st.tabs(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"])

        with tab1:
            st.header("Cluster 0 – Cluster Regular (RFM_rank 3)")
            cluster0 = rfmc[rfmc['Cluster']==0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("Recency")
                fig7, ax7 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster0['Recency'], kde=True, color='skyblue')
                ax7.set_title('Recency Distribution - Cluster 0')
                ax7.set_xlabel('Recency')
                ax7.set_ylabel('Count')
                st.pyplot(fig7)

            with col2:
                st.header("Frequency")
                fig8, ax8 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster0['Frequency'], kde=True, color='skyblue')
                ax8.set_title('Frequency Distribution - Cluster 0')
                ax8.set_xlabel('Frequency')
                ax8.set_ylabel('Count')
                st.pyplot(fig8)

            with col3:
                st.header("Monetary")
                fig9, ax9 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster0['Monetary'], kde=True, color='skyblue')
                ax9.set_title('Monetary Distribution - Cluster 0')
                ax9.set_xlabel('Monetary')
                ax9.set_ylabel('Count')
                st.pyplot(fig9)
            st.markdown("""
                - Recency 43.84 → pembelian terakhir agak lama.
                - Frequency 3.63 → jarang membeli.
                - Monetary 1.322 → nilai belanja rendah.
                - Insight: Pelanggan kurang aktif dan bernilai rendah. 
                - Strategi:
                    - Kirim reminder, voucher, atau promo khusus untuk mengembalikan mereka.
                    - Edukasi produk atau highlight benefit agar tertarik membeli lagi.
                """)
        with tab2:
            st.header("Cluster 1 – Cluster Dormant (RFM_rank 4)")
            cluster1 = rfmc[rfmc['Cluster']==1]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("Recency")
                fig7, ax7 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster1['Recency'], kde=True, color='skyblue')
                ax7.set_title('Recency Distribution - Cluster 1')
                ax7.set_xlabel('Recency')
                ax7.set_ylabel('Count')
                st.pyplot(fig7)

            with col2:
                st.header("Frequency")
                fig8, ax8 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster1['Frequency'], kde=True, color='skyblue')
                ax8.set_title('Frequency Distribution - Cluster 1')
                ax8.set_xlabel('Frequency')
                ax8.set_ylabel('Count')
                st.pyplot(fig8)

            with col3:
                st.header("Monetary")
                fig9, ax9 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster1['Monetary'], kde=True, color='skyblue')
                ax9.set_title('Monetary Distribution - Cluster 1')
                ax9.set_xlabel('Monetary')
                ax9.set_ylabel('Count')
                st.pyplot(fig9)
            st.markdown("""
                - Recency 248.54 → sudah lama tidak membeli.
                - Frequency 1.54 → hampir tidak pernah membeli.
                - Monetary 474.12 → nilai belanja sangat rendah.
                - Insight: Pelanggan sangat tidak aktif dan bernilai rendah. 
                - Strategi: 
                    - Fokus kampanye reaktivasi dengan diskon atau penawaran spesial.
                    - Evaluasi apakah perlu tetap dimasukkan dalam target pemasaran.
                """)
        with tab3:
            st.header("Cluster 2 – Cluster High Value Loyalist (RFM_rank 1)")
            cluster2 = rfmc[rfmc['Cluster']==2]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("Recency")
                fig7, ax7 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster2['Recency'], kde=True, color='skyblue')
                ax7.set_title('Recency Distribution - Cluster 2')
                ax7.set_xlabel('Recency')
                ax7.set_ylabel('Count')
                st.pyplot(fig7)

            with col2:
                st.header("Frequency")
                fig8, ax8 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster2['Frequency'], kde=True, color='skyblue')
                ax8.set_title('Frequency Distribution - Cluster 2')
                ax8.set_xlabel('Frequency')
                ax8.set_ylabel('Count')
                st.pyplot(fig8)

            with col3:
                st.header("Monetary")
                fig9, ax9 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster2['Monetary'], kde=True, color='skyblue')
                ax9.set_title('Monetary Distribution - Cluster 2')
                ax9.set_xlabel('Monetary')
                ax9.set_ylabel('Count')
                st.pyplot(fig9)
            st.markdown("""
                - Recency 7.38 → pelanggan baru-baru ini membeli.
                - Frequency 81.77 → pelanggan sering membeli.
                - Monetary 125.712 → pelanggan bernilai transaksi sangat tinggi.
                - Insight: Ini pelanggan paling loyal dan bernilai tinggi. 
                - Strategi: 
                    - Target promo eksklusif, reward, atau membership.
                    - Pertahankan loyalitas dengan program khusus atau early access produk baru.
                """)
        with tab4:
            st.header("Cluster 3 – Cluster Potensial Loyal (RFM_rank 2)")
            cluster3 = rfmc[rfmc['Cluster']==3]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("Recency")
                fig7, ax7 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster3['Recency'], kde=True, color='skyblue')
                ax7.set_title('Recency Distribution - Cluster 3')
                ax7.set_xlabel('Recency')
                ax7.set_ylabel('Count')
                st.pyplot(fig7)

            with col2:
                st.header("Frequency")
                fig8, ax8 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster3['Frequency'], kde=True, color='skyblue')
                ax8.set_title('Frequency Distribution - Cluster 3')
                ax8.set_xlabel('Frequency')
                ax8.set_ylabel('Count')
                st.pyplot(fig8)

            with col3:
                st.header("Monetary")
                fig9, ax9 = plt.subplots(figsize=(6,4))
                sns.histplot(cluster3['Monetary'], kde=True, color='skyblue')
                ax9.set_title('Monetary Distribution - Cluster 3')
                ax9.set_xlabel('Monetary')
                ax9.set_ylabel('Count')
                st.pyplot(fig9)
            st.markdown("""
                - Recency 15.39 → pembelian terakhir masih relatif baru.
                - Frequency 21.97 → frekuensi beli sedang.
                - Monetary 12.240 → nilai transaksi belanja sedang.
                - Insight: Pelanggan cukup aktif dan bernilai sedang. 
                - Strategi: 
                    - Dorong frekuensi beli dengan bundling produk atau upsell.
                    - Kirim penawaran menarik agar lebih sering bertransaksi.
                """)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Loyalist", "Big Spender", "Best Customers", "Customer Paling Baru", "At Risk Customers", "Regular Cust"])
        with tab1 :
            st.header("Loyalist")
            loyal_cust = rfmc[rfmc['Cluster'] == 2].sort_values(by='Frequency', ascending=False)
            st.dataframe(loyal_cust.head(10))
        with tab2 :
            st.header("Big Spender")
            most_spent_cust = rfmc[rfmc['Cluster'] == 2].sort_values(by='Monetary', ascending=False)
            st.dataframe(most_spent_cust.head(10))
        with tab3 :
            st.header("Best Customers")
            best_customers = rfmc.sort_values('RFM_weighted', ascending=False)
            st.dataframe(best_customers.head(10))
        with tab4 :
            st.header("Customer Paling Baru")
            newest_cust = rfmc.sort_values(
                by=['R_norm', 'F_norm'], 
                ascending=[False, True]
            )
            st.dataframe(newest_cust.head(10))
        with tab5 :
            st.header("At Risk Customers")
            at_risk_cust = rfmc.sort_values(
                by=['R_norm', 'F_norm', 'M_norm'],
                ascending=[True, False, False]
            )
            st.dataframe(at_risk_cust.head(10))
        with tab6 :
            st.header("Regular Cust")
            # Hitung kuartil
            R_low, R_high = rfmc['R_norm'].quantile([0.25, 0.75])
            F_low, F_high = rfmc['F_norm'].quantile([0.25, 0.75])
            M_low, M_high = rfmc['M_norm'].quantile([0.25, 0.75])

            # Filter customer regular
            regular_cust = rfmc[
                (rfmc['R_norm'] >= R_low) & (rfmc['R_norm'] <= R_high) &
                (rfmc['F_norm'] >= F_low) & (rfmc['F_norm'] <= F_high) &
                (rfmc['M_norm'] >= M_low) & (rfmc['M_norm'] <= M_high)
            ]
            st.dataframe(regular_cust.head(10))