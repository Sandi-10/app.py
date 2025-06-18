import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("personality_dataset.csv")  # Ganti sesuai nama file dataset Anda
    return df

df = load_data()

# Sidebar navigasi
st.sidebar.title("🧠 Aplikasi Prediksi Kepribadian")
page = st.sidebar.radio("Pilih Halaman", ["📖 Panduan", "📊 Informasi Dataset", "📈 Pemodelan Data", "🤔 Prediksi", "👥 Anggota", "🧩 Kelompok"])

# Halaman 1: Panduan
if page == "📖 Panduan":
    st.title("📖 Panduan Penggunaan Aplikasi")

    st.markdown("""
    Aplikasi ini menggunakan pembelajaran mesin untuk memprediksi tipe kepribadian seseorang berdasarkan data psikologis.

    Langkah-langkah penggunaannya:
    1. Tinjau informasi dataset (struktur data, statistik, korelasi, dan visualisasi).
    2. Latih model di halaman Pemodelan Data.
    3. Masukkan data baru di halaman Prediksi untuk melihat hasil kepribadian.

    #### ✨ Fitur yang Digunakan:
    - **Usia**
    - **Jenis Kelamin**
    - **Openness, Neuroticism, Conscientiousness, Agreeableness, Extraversion**
    - **Waktu Sendiri, Frekuensi Keluar Rumah, Merasa Lelah Setelah Bersosialisasi**
    - **Takut Panggung, Frekuensi Membuat Postingan, Ukuran Lingkaran Pertemanan**

    #### 🧠 Tentang Pemodelan:
    - **Random Forest**: Model ensemble berbasis pohon keputusan, andal untuk menangani banyak fitur dan klasifikasi multiklas.
    - **Logistic Regression**: Model linier untuk klasifikasi multiklas, berguna untuk interpretasi koefisien fitur.

    #### 💡 Saran Penggunaan:
    - Pastikan Anda telah memilih model dan mengatur parameter sebelum menekan tombol **Latih Model**.
    - Anda bisa melihat performa model melalui akurasi, confusion matrix, dan pentingnya fitur.
    - Untuk prediksi baru, isi semua input sesuai data dan tekan tombol **Prediksi**.

    *Versi saat ini menggunakan dataset bawaan dari sumber terpercaya dan tidak memungkinkan pengunggahan data pribadi.*
    """)

# Halaman 2: Informasi Dataset
elif page == "📊 Informasi Dataset":
    st.title("📊 Informasi Dataset Kepribadian")
    
    st.subheader("📄 Deskripsi Dataset")
    st.markdown("""
    Dataset ini berisi data hasil survei kepribadian berdasarkan lima dimensi besar psikologi (Big Five) dan beberapa kebiasaan sosial:
    - Openness, Neuroticism, Conscientiousness, Agreeableness, Extraversion
    - Umur, jenis kelamin, interaksi sosial, preferensi pribadi, ukuran lingkar sosial

    Target variabel adalah tipe kepribadian seperti INTP, ESFJ, dll.
    """)

    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df.describe(include='all'))

    st.subheader("📈 Distribusi Tipe Kepribadian")
    fig, ax = plt.subplots()
    df['Personality'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("🔗 Korelasi Antar Fitur")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# Halaman 3: Pemodelan Data
elif page == "📈 Pemodelan Data":
    st.title("📈 Pemodelan Kepribadian")

    st.markdown("Silakan pilih model dan parameter yang akan digunakan untuk pelatihan.")

    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    test_size = st.slider("Ukuran Data Uji (%)", 10, 50, 30)
    random_state = st.number_input("Random State", value=42)

    model_option = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"])
    
    if st.button("Latih Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)

        if model_option == "Random Forest":
            model = RandomForestClassifier(random_state=random_state)
        else:
            model = LogisticRegression(max_iter=1000, random_state=random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📄 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        if model_option == "Logistic Regression":
            st.subheader("🔢 Koefisien Fitur")
            coef_df = pd.DataFrame(model.coef_, columns=X.columns, index=model.classes_)
            st.dataframe(coef_df)

        elif model_option == "Random Forest":
            st.subheader("🌲 Pentingnya Fitur")
            importance = pd.Series(model.feature_importances_, index=X.columns)
            fig2, ax2 = plt.subplots()
            importance.sort_values().plot(kind='barh', ax=ax2)
            st.pyplot(fig2)

# Halaman 4: Prediksi
elif page == "🤔 Prediksi":
    st.title("🤔 Prediksi Kepribadian")

    st.markdown("Masukkan data berikut untuk memprediksi tipe kepribadian seseorang.")

    usia = st.slider("Usia", 12, 80, 25)
    gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    jk = 1 if gender == "Laki-laki" else 0

    openness = st.slider("Openness", 1.0, 5.0, 3.0)
    neuroticism = st.slider("Neuroticism", 1.0, 5.0, 3.0)
    conscientiousness = st.slider("Conscientiousness", 1.0, 5.0, 3.0)
    agreeableness = st.slider("Agreeableness", 1.0, 5.0, 3.0)
    extraversion = st.slider("Extraversion", 1.0, 5.0, 3.0)

    waktu_sendiri = st.slider("Waktu Sendiri (jam/hari)", 0, 24, 4)
    keluar_rumah = st.slider("Frekuensi Keluar Rumah (per minggu)", 0, 14, 3)
    lelah_sosialisasi = st.slider("Merasa Lelah Setelah Bersosialisasi (1–5)", 1, 5, 3)
    takut_panggung = st.slider("Takut Berbicara di Depan Umum (1–5)", 1, 5, 3)
    postingan = st.slider("Frekuensi Membuat Postingan (per minggu)", 0, 14, 2)
    lingkaran = st.slider("Ukuran Lingkaran Pertemanan", 0, 100, 10)

    input_data = pd.DataFrame([[usia, jk, openness, neuroticism, conscientiousness, agreeableness,
                                extraversion, waktu_sendiri, keluar_rumah, lelah_sosialisasi,
                                takut_panggung, postingan, lingkaran]],
                              columns=['Usia', 'Jenis Kelamin', 'Openness', 'Neuroticism',
                                       'Conscientiousness', 'Agreeableness', 'Extraversion',
                                       'Waktu Sendiri', 'Frekuensi Keluar Rumah',
                                       'Merasa Lelah Setelah Bersosialisasi',
                                       'Takut Panggung', 'Frekuensi Membuat Postingan',
                                       'Ukuran Lingkaran Pertemanan'])

    if st.button("Prediksi"):
        model = RandomForestClassifier().fit(df.drop("Personality", axis=1), df["Personality"])
        hasil = model.predict(input_data)[0]
        st.success(f"Tipe Kepribadian yang Diprediksi: {hasil}")

        st.markdown("""
        Tipe kepribadian ini diprediksi berdasarkan pola data yang Anda masukkan.
        Jika Anda ingin hasil lebih akurat, silakan masukkan data yang lengkap dan representatif.
        """)

# Halaman tambahan dummy
elif page == "👥 Anggota":
    st.title("👥 Anggota")
    st.markdown("Aplikasi ini dikembangkan oleh tim ...")

elif page == "🧩 Kelompok":
    st.title("🧩 Kelompok")
    st.markdown("Bagian ini menjelaskan pembagian kelompok kepribadian berdasarkan teori MBTI.")
