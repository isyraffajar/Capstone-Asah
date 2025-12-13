import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI PATH (GANTI BAGIAN INI) ---

FILE_MODEL = 'Code/model_clustering.pkl'
FILE_SCALER = 'Code/scaler.pkl'
FILE_CSV_DATA = 'Dataset/rfm_with_clusters.csv' 

# --- 1. Load Model, Scaler & Data ---
@st.cache_resource
def load_resources():
    try:
        # Load Model & Scaler langsung dari path yang ditentukan di atas
        model = joblib.load(FILE_MODEL)
        scaler = joblib.load(FILE_SCALER)
        
        # Load Data RFM Asli (untuk visualisasi histogram)
        try:
            rfm_data = pd.read_csv(FILE_CSV_DATA)
        except FileNotFoundError:
            # Data Dummy untuk backup jika CSV tidak ditemukan (agar app tidak crash)
            st.warning(f"File '{FILE_CSV_DATA}' tidak ditemukan. Menggunakan data dummy untuk visualisasi.")
            rfm_data = pd.DataFrame({
                'Recency': np.random.randint(1, 365, 100),
                'Frequency': np.random.randint(1, 50, 100),
                'Monetary': np.random.randint(1000, 100000, 100),
                'Cluster': np.random.randint(0, 3, 100)
            })
            
        return model, scaler, rfm_data
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        st.error("Pastikan path/alamat file di bagian atas kode sudah benar.")
        return None, None, None

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Customer Segmentation App", page_icon="🛍️", layout="wide")

model, scaler, rfmc = load_resources()

# --- 2. Judul & Sidebar ---
st.title("🛍️ Prediksi Segmentasi Pelanggan")
st.markdown("Masukkan data transaksi pelanggan untuk mengetahui segmen dan strategi marketing yang tepat.")

st.sidebar.header("📝 Input Data Pelanggan")
recency = st.sidebar.number_input("Recency (Hari sejak pembelian terakhir)", min_value=0, value=30)
frequency = st.sidebar.number_input("Frequency (Jumlah transaksi)", min_value=1, value=5)
monetary = st.sidebar.number_input("Monetary (Total belanja)", min_value=0.0, value=100000.0, step=1000.0)

input_df = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})

# --- 3. Proses Prediksi & Tampilan Hasil ---
if st.button("🔍 Analisis Segmen"):
    if model is not None and scaler is not None:
        # Prediksi
        data_scaled = scaler.transform(input_df)
        cluster_prediction = model.predict(data_scaled)
        cluster_id = cluster_prediction[0]

        st.divider()
        st.success(f"### Hasil Prediksi: Pelanggan Masuk ke Cluster {cluster_id}")

        # Filter data historis berdasarkan cluster hasil prediksi
        if 'Cluster' in rfmc.columns:
            cluster_data = rfmc[rfmc['Cluster'] == cluster_id]
        else:
            cluster_data = rfmc # Fallback jika kolom cluster tidak ada

        st.subheader(f"📊 Analisis Detail Cluster {cluster_id}")
        
        # Tampilkan Histogram
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Distribusi Recency**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(cluster_data['Recency'], kde=True, color='skyblue', ax=ax1)
            ax1.axvline(recency, color='red', linestyle='--', label='Input User')
            ax1.legend()
            ax1.set_title(f'Recency Cluster {cluster_id}')
            st.pyplot(fig1)

        with col2:
            st.write("**Distribusi Frequency**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(cluster_data['Frequency'], kde=True, color='lightgreen', ax=ax2)
            ax2.axvline(frequency, color='red', linestyle='--', label='Input User')
            ax2.legend()
            ax2.set_title(f'Frequency Cluster {cluster_id}')
            st.pyplot(fig2)

        with col3:
            st.write("**Distribusi Monetary**")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.histplot(cluster_data['Monetary'], kde=True, color='salmon', ax=ax3)
            ax3.axvline(monetary, color='red', linestyle='--', label='Input User')
            ax3.legend()
            ax3.set_title(f'Monetary Cluster {cluster_id}')
            st.pyplot(fig3)

        # --- Strategi & Insight ---
        st.divider()
        st.subheader(f"💡 Insight & Strategi untuk Cluster {cluster_id}")

        if cluster_id == 0:
            st.info("**Cluster 0** – Cluster Regular (RFM_rank 3)")
            st.markdown(f"""
            - Recency {cluster_data['Recency'].mean():.2f} → pembelian terakhir agak lama.
            - Frequency {cluster_data['Frequency'].mean():.2f} → jarang membeli.
            - Monetary {cluster_data['Monetary'].mean():.2f} → nilai belanja rendah.
            - Insight: Pelanggan kurang aktif dan bernilai rendah. 
            - Strategi:
                - Kirim reminder, voucher, atau promo khusus untuk mengembalikan mereka.
                - Edukasi produk atau highlight benefit agar tertarik membeli lagi.
            """)
        
        elif cluster_id == 1:
            st.info("**Cluster 1** – Cluster Dormant (RFM_rank 4)")
            st.markdown(f"""
            - Recency {cluster_data['Recency'].mean():.2f} → sudah lama tidak membeli.
            - Frequency {cluster_data['Frequency'].mean():.2f} → hampir tidak pernah membeli.
            - Monetary {cluster_data['Monetary'].mean():,.0f} → nilai belanja sangat rendah.
            - Insight: Pelanggan sangat tidak aktif dan bernilai rendah. 
            - Strategi: 
                - Fokus kampanye reaktivasi dengan diskon atau penawaran spesial.
                - Evaluasi apakah perlu tetap dimasukkan dalam target pemasaran.
            """)

        elif cluster_id == 2:
            st.header("**Cluster 2** – Cluster High Value Loyalist (RFM_rank 1)")
            st.markdown(f"""
            - Recency {cluster_data['Recency'].mean():.2f} → pelanggan baru-baru ini membeli.
            - Frequency {cluster_data['Frequency'].mean():.2f} → pelanggan sering membeli.
            - Monetary {cluster_data['Monetary'].mean():,.0f} → pelanggan bernilai transaksi sangat tinggi.
            - Insight: Ini pelanggan paling loyal dan bernilai tinggi. 
            - Strategi: 
                - Target promo eksklusif, reward, atau membership.
                - Pertahankan loyalitas dengan program khusus atau early access produk baru.
            """)
        
        elif cluster_id == 3:
            st.info("**Cluster 3** – Cluster Potensial Loyal (RFM_rank 2)")
            st.markdown(f"""
            - Recency {cluster_data['Recency'].mean():.2f} → pembelian terakhir masih relatif baru.
            - Frequency {cluster_data['Frequency'].mean():.2f} → frekuensi beli sedang.
            - Monetary {cluster_data['Monetary'].mean():,.0f} → nilai transaksi belanja sedang.
            - Insight: Pelanggan cukup aktif dan bernilai sedang. 
            - Strategi: 
                - Dorong frekuensi beli dengan bundling produk atau upsell.
                - Kirim penawaran menarik agar lebih sering bertransaksi.
            """)
            
        else:
            st.write(f"Cluster {cluster_id}")
            st.write(cluster_data.describe())

    else:
        st.error("Model belum siap. Cek path file di bagian atas kode.")