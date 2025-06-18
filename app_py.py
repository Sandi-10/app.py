import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Load data
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# Rename kolom ke Bahasa Indonesia
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

# Encode target
target_encoder = LabelEncoder()
df['Kepribadian'] = target_encoder.fit_transform(df['Personality'])
df.drop(columns=['Personality'], inplace=True)

# Session State
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'rf_accuracy' not in st.session_state:
    st.session_state.rf_accuracy = None
if 'lr_accuracy' not in st.session_state:
    st.session_state.lr_accuracy = None

# Navigasi
st.sidebar.title("Navigasi Aplikasi")
page = st.sidebar.radio("Pilih Halaman", ["📖 Petunjuk", "📘 Informasi", "📊 Pemodelan Data", "🔮 Prediksi", "👥 Anggota Kelompok"])

# ============================ PETUNJUK ============================
if page == "📖 Petunjuk":
    st.title("📖 Petunjuk Penggunaan Aplikasi")
    st.markdown("""
    Selamat datang di Aplikasi Prediksi Tipe Kepribadian!

    **Fitur aplikasi ini meliputi:**
    - Visualisasi data (statistik deskriptif, distribusi, korelasi)
    - Pemodelan menggunakan 2 algoritma: Random Forest & Logistic Regression
    - Formulir interaktif untuk prediksi tipe kepribadian berdasarkan input pengguna
    - Perbandingan performa antar model

    **Langkah-langkah:**
    1. Buka halaman '📘 Informasi' untuk eksplorasi dataset
    2. Gunakan halaman '📊 Pemodelan Data' untuk melatih model
    3. Coba fitur '🔮 Prediksi' untuk melakukan prediksi baru
    4. Lihat informasi tim di '👥 Anggota Kelompok'

    Dataset digunakan langsung dari repositori GitHub, tidak diunggah secara manual.
    """)

# ============================ INFORMASI ============================
elif page == "📘 Informasi":
    st.title("📘 Informasi Dataset Kepribadian")
    st.write("Dataset ini berisi informasi tentang kepribadian berdasarkan berbagai faktor psikologis.")

    st.subheader("Contoh Data")
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

    st.subheader("Boxplot Setiap Fitur Numerik")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col != 'Kepribadian':
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Kepribadian', y=col, ax=ax)
            ax.set_title(f"Distribusi {col} berdasarkan Kepribadian")
            ax.set_xticklabels(target_encoder.inverse_transform(sorted(df['Kepribadian'].unique())))
            st.pyplot(fig)

# ============================ PEMODELAN ============================
elif page == "📊 Pemodelan Data":
    st.title("📊 Pemodelan Prediksi Kepribadian")

    df_model = df.copy()
    X = df_model.drop('Kepribadian', axis=1)
    y = df_model['Kepribadian']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Pilih Model dan Parameter")
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])

    if model_choice == "Random Forest":
        n_estimators = st.slider("Jumlah Pohon (n_estimators)", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == "Logistic Regression":
        max_iter = st.slider("Jumlah Iterasi Maksimum", 100, 500, 200)
        model = LogisticRegression(max_iter=max_iter, solver='lbfgs')

    if st.button("🚀 Latih Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        if model_choice == "Random Forest":
            st.session_state.rf_accuracy = acc
        else:
            st.session_state.lr_accuracy = acc

        st.metric("Akurasi", f"{acc:.2f}")

        st.subheader("📋 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_encoder.classes_,
                    yticklabels=target_encoder.classes_,
                    ax=ax_cm)
        ax_cm.set_xlabel('Prediksi')
        ax_cm.set_ylabel('Aktual')
        st.pyplot(fig_cm)

        if hasattr(model, 'feature_importances_'):
            st.subheader("📌 Pentingnya Fitur")
            importance = model.feature_importances_
            imp_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importance}).sort_values(by='Pentingnya', ascending=False)
            fig_imp, ax_imp = plt.subplots()
            sns.barplot(x='Pentingnya', y='Fitur', data=imp_df, palette='viridis', ax=ax_imp)
            st.pyplot(fig_imp)

        if len(target_encoder.classes_) == 2:
            st.subheader("🚦 ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax3 = plt.subplots()
            ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax3.set_title("ROC Curve")
            ax3.set_xlabel("False Positive Rate")
            ax3.set_ylabel("True Positive Rate")
            ax3.legend()
            st.pyplot(fig_roc)

    # Visualisasi Perbandingan Model
    if st.session_state.rf_accuracy is not None and st.session_state.lr_accuracy is not None:
        st.subheader("📊 Perbandingan Akurasi Dua Model")
        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'Logistic Regression'],
            'Akurasi': [st.session_state.rf_accuracy, st.session_state.lr_accuracy]
        })
        fig_cmp, ax_cmp = plt.subplots()
        sns.barplot(data=comparison_df, x='Model', y='Akurasi', palette='pastel', ax=ax_cmp)
        ax_cmp.set_ylim(0, 1)
        ax_cmp.set_title("Perbandingan Akurasi")
        st.pyplot(fig_cmp)

# ============================ PREDIKSI ============================
elif page == "🔮 Prediksi":
    st.title("🔮 Prediksi Tipe Kepribadian")

    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan ke halaman Pemodelan.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Kepribadian':
                if df[col].dtype in [np.float64, np.int64]:
                    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    val = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
                input_data[col] = val

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

# ============================ ANGGOTA ============================
elif page == "👥 Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    st.markdown("""
    - 👩‍🏫 Diva Auliya Pusparini (2304030041)  
    - 👩‍🎓 Paskalia Kanicha Mardian (2304030062)  
    - 👨‍💻 Sandi Krisna Mukti (2304030074)  
    - 👩‍⚕️ Siti Maisyaroh (2304030079)
    """)
