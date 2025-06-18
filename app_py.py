import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Dashboard Prediksi Kepribadian", layout="wide")

# Muat dataset
@st.cache_data
def load_data():
    df = pd.read_csv("personality_dataset.csv")
    return df

df = load_data()

# Label encoding target
le = LabelEncoder()
df["Kepribadian"] = le.fit_transform(df["Kepribadian"])

# Fitur dan target
X = df.drop("Kepribadian", axis=1)
y = df["Kepribadian"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Navigasi
halaman = st.sidebar.radio("Pilih Halaman", ["Informasi", "Pemodelan Data", "Perbandingan Model", "Prediksi Manual"])

# Halaman Informasi
if halaman == "Informasi":
    st.title("📘 Informasi Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk melakukan prediksi tipe kepribadian seseorang berdasarkan data survei.

    ### Dataset
    Dataset ini berisi beberapa fitur yang merepresentasikan preferensi individu dalam aktivitas sosial dan pribadi:

    - **Merasa Lebih Sering Dimengerti**
    - **Waktu Sendiri**
    - **Takut Panggung**
    - **Preferensi Membaca Daripada Pesta**
    - **Frekuensi Menghadiri Acara Sosial**
    - **Preferensi Menonton Sendiri**
    - **Liburan: Lingkungan Ramai vs Tenang**

    Target klasifikasinya adalah kepribadian: **Introvert** atau **Ekstrovert**.
    
    ### Model
    Dua model yang digunakan:
    - Random Forest Classifier
    - Logistic Regression

    Silakan navigasi ke tab lainnya untuk melihat pelatihan, evaluasi model, dan prediksi manual.
    """)

# Halaman Pemodelan Data
elif halaman == "Pemodelan Data":
    st.title("🤖 Pemodelan Data")

    model_pilihan = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"])

    if model_pilihan == "Random Forest":
        n_estimators = st.slider("Jumlah Pohon (n_estimators)", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        C_value = st.slider("Nilai C (Regulasi)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C_value, solver="liblinear")

    if st.button("⚙️ Latih Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        st.subheader(f"🎯 Akurasi: {acc:.2f}")

        st.subheader("📊 Classification Report")
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        st.markdown("Classification Report menunjukkan metrik evaluasi seperti precision, recall, dan f1-score untuk tiap kelas.")

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        st.pyplot(plt.gcf())
        st.markdown("Confusion Matrix menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas.")

        if model_pilihan == "Random Forest":
            st.subheader("🌟 Pentingnya Fitur")
            importances = model.feature_importances_
            fitur_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            fitur_imp.plot(kind='barh')
            plt.xlabel("Pentingnya")
            st.pyplot(plt.gcf())
            st.markdown("Visualisasi ini menunjukkan kontribusi relatif dari setiap fitur terhadap model Random Forest.")

        st.subheader("🔍 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(plt.gcf())
        st.markdown("ROC Curve menunjukkan trade-off antara sensitivitas dan spesifisitas. AUC mendekati 1 menunjukkan performa baik.")

# Halaman Perbandingan Model
elif halaman == "Perbandingan Model":
    st.title("📈 Perbandingan Model")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, solver="liblinear")
    }

    akurasi = {}
    for nama, mdl in models.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        akurasi[nama] = accuracy_score(y_test, pred)

    df_akurasi = pd.DataFrame.from_dict(akurasi, orient="index", columns=["Akurasi"])
    st.bar_chart(df_akurasi)
    st.markdown("Visualisasi ini memperlihatkan perbandingan performa dua model dari segi akurasi.")

# Halaman Prediksi Manual
elif halaman == "Prediksi Manual":
    st.title("🧪 Prediksi Kepribadian Manual")

    st.markdown("Masukkan nilai untuk masing-masing fitur berikut untuk memprediksi apakah seseorang termasuk Ekstrovert atau Introvert:")

    fitur_input = {}
    for kolom in X.columns:
        fitur_input[kolom] = st.slider(f"{kolom}", float(df[kolom].min()), float(df[kolom].max()), float(df[kolom].mean()))

    fitur_df = pd.DataFrame([fitur_input])

    model_pilihan_prediksi = st.selectbox("Pilih Model untuk Prediksi", ["Random Forest", "Logistic Regression"], key="prediksi_model")

    if st.button("🔮 Prediksi Sekarang"):
        if model_pilihan_prediksi == "Random Forest":
            model_prediksi = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model_prediksi = LogisticRegression(C=1.0, solver="liblinear")

        model_prediksi.fit(X_train, y_train)
        hasil_prediksi = model_prediksi.predict(fitur_df)[0]
        hasil_label = le.inverse_transform([hasil_prediksi])[0]
        st.success(f"Hasil Prediksi: **{hasil_label}**")

