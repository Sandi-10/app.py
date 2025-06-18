import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ===================== Informasi Aplikasi =====================
st.set_page_config(page_title="Prediksi Kepribadian", layout="wide")
st.sidebar.image("https://img.icons8.com/ios-filled/100/psychology.png", width=80)
st.sidebar.title("🧠 Aplikasi Prediksi Kepribadian")

# ===================== Load Data =====================
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

df.rename(columns={
    'Age': 'Usia',
    'Gender': 'Jenis_Kelamin',
    'Openness': 'Keterbukaan',
    'Neuroticism': 'Neurotisisme',
    'Conscientiousness': 'Kehati_hatian',
    'Agreeableness': 'Sifat_Mudah_Setuju',
    'Extraversion': 'Ekstraversi',
    'Time_spent_Alone': 'Waktu_Sendiri',
    'Stage_fear': 'Takut_Panggung',
    'Social_event_attendance': 'Frekuensi Menghadiri Acara Sosial',
    'Going_outside': 'Frekuensi Keluar Rumah',
    'Drained_after_socializing': 'Merasa Lelah Setelah Bersosialisasi',
    'Friends_circle_size': 'Ukuran Lingkaran Pertemanan',
    'Post_frequency': 'Frekuensi Membuat Postingan',
}, inplace=True)

target_encoder = LabelEncoder()
df['Kepribadian'] = target_encoder.fit_transform(df['Personality'])
df.drop(columns=['Personality'], inplace=True)

# ===================== Session State =====================
for key in ['model', 'X_columns', 'X_test', 'y_test']:
    if key not in st.session_state:
        st.session_state[key] = None

# ===================== Navigasi =====================
page = st.sidebar.radio("Pilih Halaman", [
    "📖 Panduan",
    "📘 Informasi Dataset", 
    "📊 Pemodelan Data", 
    "🔮 Prediksi", 
    "👥 Anggota Kelompok"
])

# ===================== Panduan =====================
if page == "📖 Panduan":
    st.title("📖 Panduan Penggunaan Aplikasi Prediksi Kepribadian")
    st.markdown("""
Aplikasi ini dirancang untuk memprediksi tipe kepribadian seseorang berdasarkan fitur psikologis dan sosial yang dimilikinya.

#### ✨ Fitur yang Digunakan:
- *Usia*: Umur responden.
- *Jenis Kelamin*: Laki-laki atau perempuan.
- *Keterbukaan (Openness), Neurotisisme, Kehati-hatian (Conscientiousness), Sifat Mudah Setuju (Agreeableness), Ekstraversi*: Lima dimensi kepribadian utama.
- *Waktu Sendiri, Frekuensi Keluar Rumah, Merasa Lelah Setelah Bersosialisasi*: Indikator preferensi sosial.
- *Takut Panggung, Frekuensi Membuat Postingan, Ukuran Lingkaran Pertemanan*: Ciri sosial lainnya.

#### 🧠 Tentang Pemodelan:
- *Random Forest*: Model ensemble berbasis pohon keputusan, andal dan mampu menangani banyak fitur sekaligus.
- *Logistic Regression*: Model linier untuk klasifikasi multiklas, berguna untuk interpretasi bobot/koefisien fitur.

#### 🧾 Saran Penggunaan:
- Pastikan Anda telah memilih model dan mengatur parameter sebelum menekan tombol *Latih Model*.
- Anda bisa melihat performa model melalui akurasi, confusion matrix, dan pentingnya fitur.
- Untuk prediksi baru, isi semua input sesuai data dan tekan tombol *Prediksi*.

> Versi saat ini menggunakan dataset bawaan dari sumber terpercaya dan tidak memungkinkan pengunggahan data eksternal.
""")

# ===================== Informasi Dataset =====================
elif page == "📘 Informasi Dataset":
    st.title("📘 Informasi Dataset Kepribadian")
    st.dataframe(df.head())

    st.subheader("Deskripsi Statistik")
    st.write(df.describe(include='all'))

    st.subheader("Distribusi Tipe Kepribadian")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Kepribadian', ax=ax1)
    ax1.set_xticklabels(target_encoder.inverse_transform(sorted(df['Kepribadian'].unique())))
    st.pyplot(fig1)

    st.subheader("Korelasi Antar Fitur")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# ===================== Pemodelan =====================
elif page == "📊 Pemodelan Data":
    st.title("📊 Pemodelan Prediksi Kepribadian")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        st.success("✅ Nilai kosong/∞ berhasil diimputasi.")
    else:
        st.success("✅ Tidak ada nilai kosong/∞.")

    X = df.drop('Kepribadian', axis=1)
    y = df['Kepribadian']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Pilih Model")
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])

    if model_choice == "Random Forest":
        n_estimators = st.slider("Jumlah Pohon", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        model = LogisticRegression(max_iter=500)

    if st.button("🚀 Latih Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        acc = accuracy_score(y_test, y_pred)
        st.metric("Akurasi Data Uji", f"{acc:.2f}")

        with st.spinner("Melakukan Cross-Validation..."):
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            st.metric("Akurasi Rata-rata (CV)", f"{cv_scores.mean():.2f}")

        st.subheader("📋 Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)).transpose().style.format("{:.2f}"))

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_encoder.classes_,
                    yticklabels=target_encoder.classes_,
                    ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

        if hasattr(model, 'feature_importances_'):
            st.subheader("📌 Pentingnya Fitur")
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'Fitur': X.columns, 'Penting': importances})
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Penting', y='Fitur', data=imp_df.sort_values(by='Penting', ascending=False), palette='viridis', ax=ax2)
            st.pyplot(fig2)

        if isinstance(model, LogisticRegression):
            st.subheader("📉 Koefisien Fitur (Logistic Regression)")
            coef_df = pd.DataFrame({
                'Fitur': X.columns,
                'Koefisien': model.coef_[0]
            }).sort_values(by='Koefisien', ascending=False)
            fig3, ax3 = plt.subplots()
            sns.barplot(x='Koefisien', y='Fitur', data=coef_df, palette='coolwarm', ax=ax3)
            st.pyplot(fig3)

# ===================== Prediksi =====================
elif page == "🔮 Prediksi":
    st.title("🔮 Prediksi Tipe Kepribadian")
    if st.session_state.model is None:
        st.warning("Latih model terlebih dahulu di halaman Pemodelan.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Kepribadian':
                if df[col].dtype in [np.float64, np.int64]:
                    input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    input_data[col] = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
        input_df = pd.DataFrame([input_data])

        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(df[col])
                input_df[col] = le.transform(input_df[col])

        input_df = input_df[st.session_state.X_columns]

        if st.button("Prediksi"):
            pred = st.session_state.model.predict(input_df)[0]
            prob = st.session_state.model.predict_proba(input_df)[0]
            label = target_encoder.inverse_transform([pred])[0]
            st.success(f"Tipe Kepribadian yang Diprediksi: {label}")
            st.subheader("Probabilitas")
            st.bar_chart(pd.Series(prob, index=target_encoder.classes_))

# ===================== Anggota =====================
elif page == "👥 Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    st.markdown("""
- 👩‍🏫 Diva Auliya Pusparini (2304030041)  
- 👩‍🎓 Paskalia Kanicha Mardian (2304030062)  
- 👨‍💻 Sandi Krisna Mukti (2304030074)  
- 👩‍⚕ Siti Maisyaroh (2304030079)
""")
