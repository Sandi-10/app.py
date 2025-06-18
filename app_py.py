import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# ===================== Load Data =====================
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

# Session state
for key in ['model', 'X_columns', 'X_test', 'y_test']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar Navigasi
st.sidebar.title("Navigasi Aplikasi")
page = st.sidebar.radio("Pilih Halaman", [
    "📖 Panduan",
    "📌 Petunjuk Penggunaan", 
    "📘 Informasi Dataset", 
    "📊 Pemodelan Data", 
    "🔮 Prediksi", 
    "👥 Anggota Kelompok"
])

# ============================ PANDUAN ============================
if page == "📖 Panduan":
    st.title("📖 Panduan Penggunaan Aplikasi Prediksi Kepribadian")
    st.markdown("""
    Aplikasi ini dirancang untuk memprediksi tipe kepribadian seseorang berdasarkan fitur psikologis.
    Masuk ke menu Pemodelan untuk melatih model dan ke halaman Prediksi untuk mencoba prediksi berdasarkan input Anda.
    """)

# ============================ PETUNJUK ============================
elif page == "📌 Petunjuk Penggunaan":
    st.title("📌 Petunjuk Penggunaan Aplikasi")
    st.markdown("""
    Berikut adalah langkah-langkah menggunakan aplikasi ini:

    1. Buka halaman "📘 Informasi Dataset" untuk melihat data dan statistik dasar.
    2. Buka halaman "📊 Pemodelan Data" untuk melatih model prediksi.
    3. Setelah model dilatih, buka halaman "🔮 Prediksi" untuk memprediksi kepribadian berdasarkan data input Anda.
    """)

# ============================ INFORMASI ============================
elif page == "📘 Informasi Dataset":
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

    st.subheader("🔍 Validasi Data")
    if st.button("🔍 Cek Validitas Data"):
        nan_X = X.isnull().sum()
        nan_y = y.isnull().sum()
        inf_X = np.isinf(X).sum()
        inf_y = np.isinf(y).sum()

        if nan_X.sum() > 0 or nan_y > 0:
            st.error("❗ Ditemukan nilai kosong (NaN) dalam data.")
            st.write("Detail NaN di fitur (X):")
            st.write(nan_X[nan_X > 0])
            if nan_y > 0:
                st.write("Jumlah NaN di target (y):", nan_y)
        elif inf_X.sum() > 0 or inf_y > 0:
            st.error("❗ Ditemukan nilai tak hingga (∞) dalam data.")
            st.write("Jumlah ∞ di fitur (X):", inf_X[inf_X > 0])
        else:
            st.success("✅ Tidak ditemukan NaN atau ∞. Data aman untuk proses pelatihan.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Pilih Model dan Parameter")
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])

    if model_choice == "Random Forest":
        n_estimators = st.slider("Jumlah Pohon (n_estimators)", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=500)

    if st.button("🚀 Latih Model"):
        if (
            np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)) or
            np.any(np.isinf(X_train)) or np.any(np.isinf(y_train))
        ):
            st.error("❌ Data pelatihan mengandung NaN atau ∞. Harap bersihkan data sebelum melatih model.")
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)

            st.session_state.model = model
            st.session_state.X_columns = X.columns.tolist()
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.metric("Akurasi Data Uji", f"{acc:.2f}")

            with st.spinner("Melakukan Cross-Validation..."):
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                st.metric("Cross-Validation Akurasi (rata-rata)", f"{cv_scores.mean():.2f}")
                st.write("Akurasi per Fold:", [f"{score:.2f}" for score in cv_scores])

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
                imp_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importance})
                fig_imp, ax_imp = plt.subplots()
                sns.barplot(x='Pentingnya', y='Fitur', data=imp_df.sort_values(by='Pentingnya', ascending=False), palette='viridis', ax=ax_imp)
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
