import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar
st.sidebar.title("🧠 Aplikasi Prediksi Kepribadian")
page = st.sidebar.radio("Pilih Halaman", [
    "📖 Panduan",
    "📊 Informasi Dataset",
    "🧪 Pemodelan Data",
    "🔍 Prediksi",
    "👥 Anggota",
    "🧑 Kelompok"
])

# Halaman Panduan
if page == "📖 Panduan":
    st.title("📖 Panduan Penggunaan Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan pembelajaran mesin untuk memprediksi tipe kepribadian seseorang berdasarkan data psikologis.
    
    Langkah-langkah penggunaannya:
    1. Tinjau informasi dataset (struktur data, statistik, korelasi, dan visualisasi).
    2. Latih model di halaman Pemodelan Data.
    3. Masukkan data baru di halaman Prediksi untuk melihat hasil kepribadian.

    #### ✨ Fitur yang Digunakan:
    - **Usia**: Umur responden.
    - **Jenis Kelamin**: Laki-laki atau perempuan.
    - **Keterbukaan (Openness)**, **Neurotisisme**, **Kehati-hatian (Conscientiousness)**, **Sifat Ramah (Agreeableness)**, **Ekstraversi**.
    - **Waktu Sendiri**, **Frekuensi Keluar Rumah**, **Merasa Lelah Setelah Bersosialisasi**, **Takut Panggung**, **Frekuensi Membuat Postingan**, **Ukuran Lingkaran Pertemanan**.

    #### 🤖 Tentang Pemodelan:
    - **Random Forest**: Model ensemble berbasis pohon keputusan yang mampu menangani data kompleks dan interaksi fitur.
    - **Logistic Regression**: Model linier untuk klasifikasi multiklas, berguna untuk interpretasi koefisien fitur.

    #### 📌 Penjelasan Tambahan Hasil Model:
    - **Classification Report**: Menyajikan metrik seperti precision, recall, f1-score untuk masing-masing kelas tipe kepribadian.
    - **Confusion Matrix**: Matriks yang menunjukkan prediksi benar dan salah antar tipe. Membantu melihat pola kesalahan model.
    - **Koefisien Fitur (Logistic Regression)**: Menunjukkan kekuatan dan arah pengaruh masing-masing fitur terhadap prediksi kelas.
    - **Feature Importance (Random Forest)**: Memberi tahu fitur mana yang paling berpengaruh dalam prediksi model.

    #### 📌 Saran Penggunaan:
    - Pastikan Anda telah memilih model dan mengatur parameter sebelum menekan tombol **Latih Model**.
    - Anda bisa melihat performa model melalui akurasi, confusion matrix, dan pentingnya fitur.
    - Untuk prediksi baru, isi semua input sesuai data dan tekan tombol **Prediksi**.

    *Versi saat ini menggunakan dataset bawaan dari sumber terpercaya dan tidak memungkinkan pengguna mengunggah data sendiri.*
    """)

# Halaman Informasi Dataset
elif page == "📊 Informasi Dataset":
    st.title("📊 Informasi Dataset")
    df = pd.read_csv("dataset_kepribadian.csv")  # Pastikan file ini tersedia di direktori
    st.subheader("🔍 Deskripsi Dataset")
    st.write("Dataset ini berisi data psikologis responden untuk memprediksi tipe kepribadian mereka.")

    st.markdown("#### 📈 Statistik Deskriptif")
    st.dataframe(df.describe())

    st.markdown("#### 📊 Distribusi Tipe Kepribadian")
    st.bar_chart(df["Kepribadian"].value_counts())

    st.markdown("#### 🔗 Korelasi Antar Fitur")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    **Penjelasan Singkat**:
    - Statistik deskriptif membantu memahami rentang nilai dan penyebaran data.
    - Distribusi kepribadian menunjukkan proporsi responden dalam tiap kategori MBTI.
    - Korelasi berguna untuk melihat hubungan antar fitur, misalnya apakah usia berkorelasi dengan tingkat ekstraversi.
    """)

# Halaman Pemodelan dan lainnya bisa ditambahkan berikutnya
