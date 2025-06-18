import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Halaman Informasi Aplikasi ---
st.set_page_config(page_title="Prediksi Kepribadian", layout="wide")
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman", ["Informasi", "Pemodelan Data", "Perbandingan Model"])

if halaman == "Informasi":
    st.title("📘 Informasi Aplikasi Prediksi Kepribadian")
    st.markdown("""
    Aplikasi ini dirancang untuk memprediksi kepribadian seseorang (Ekstrovert atau Introvert) berdasarkan beberapa faktor seperti:

    - Merasa Lebih Bersemangat Saat Bersama Orang Lain
    - Waktu Sendiri
    - Takut Panggung
    - Frekuensi Bertemu Teman
    - Preferensi Menghindari Acara Sosial
    - Frekuensi Percakapan Sehari-hari
    - Literasi tentang Preferensi Kepribadian

    **Model yang digunakan**:
    - Random Forest
    - Logistic Regression

    Di halaman pemodelan, Anda dapat melatih model dan melihat evaluasi seperti:
    - Classification Report
    - Confusion Matrix
    - Pentingnya Fitur
    - ROC Curve

    Serta membandingkan performa kedua model di halaman Perbandingan Model.
    """)

# --- Load dataset ---
df = pd.read_csv("personality_dataset (1).csv")
df = df.dropna()

# Label encoding target
label = LabelEncoder()
df["Kepribadian"] = label.fit_transform(df["Kepribadian"])

X = df.drop("Kepribadian", axis=1)
y = df["Kepribadian"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if halaman == "Pemodelan Data":
    st.title("🔬 Pemodelan Data Kepribadian")

    model_terpilih = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"])

    if model_terpilih == "Random Forest":
        n_estimators = st.slider("Jumlah Pohon (n_estimators)", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_terpilih == "Logistic Regression":
        c_value = st.slider("Nilai C (Regulasi)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=c_value, solver="liblinear")

    if st.button("🚀 Latih Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        st.subheader("🎯 Akurasi")
        st.write(f"{acc:.2f}")

        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, target_names=["Ekstrovert", "Introvert"])
        st.dataframe(pd.DataFrame(report).transpose())
        st.markdown("""
        **Penjelasan**: Classification Report memberikan metrik evaluasi seperti precision, recall, dan f1-score untuk masing-masing kelas kepribadian. 
        Nilai-nilai ini membantu mengevaluasi seberapa baik model mengenali tiap kategori.
        """)

        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ekstrovert", "Introvert"], yticklabels=["Ekstrovert", "Introvert"])
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        st.pyplot(fig)
        st.markdown("""
        **Penjelasan**: Confusion Matrix menunjukkan jumlah prediksi benar dan salah yang dibuat oleh model untuk masing-masing kelas.
        """)

        # Feature Importance
        if model_terpilih == "Random Forest":
            st.subheader("📌 Pentingnya Fitur")
            feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig2, ax2 = plt.subplots()
            sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax2)
            plt.xlabel("Pentingnya")
            plt.ylabel("Fitur")
            st.pyplot(fig2)
            st.markdown("""
            **Penjelasan**: Visualisasi ini menunjukkan fitur mana yang paling berkontribusi terhadap prediksi model Random Forest.
            """)

        # ROC Curve
        st.subheader("🔍 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig3, ax3 = plt.subplots()
        ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax3.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(fig3)
        st.markdown("""
        **Penjelasan**: ROC Curve membantu mengevaluasi kinerja model secara keseluruhan dalam memisahkan dua kelas. Semakin tinggi AUC, semakin baik model.
        """)

# --- Halaman Perbandingan Model ---
elif halaman == "Perbandingan Model":
    st.title("⚖️ Perbandingan Dua Model")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(C=1.0, solver="liblinear")

    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    st.subheader("Perbandingan Akurasi")
    acc_df = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "Akurasi": [acc_rf, acc_lr]
    })

    fig4, ax4 = plt.subplots()
    sns.barplot(x="Model", y="Akurasi", data=acc_df, ax=ax4)
    plt.ylim(0, 1)
    st.pyplot(fig4)
    st.markdown("""
    **Penjelasan**: Visualisasi ini menunjukkan perbandingan performa akurasi antara Random Forest dan Logistic Regression.
    Anda dapat menggunakan ini untuk menentukan model terbaik untuk digunakan.
    """)
