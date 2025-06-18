import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import base64

# Fungsi load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("personality_dataset.csv")
    return df

# Fungsi encode gambar background (jika digunakan)
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load data
df = load_data()

# Persiapan data
X = df.drop(columns=["Personality (Class label)"])
y = df["Personality (Class label)"]

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Informasi Dataset", "Pemodelan", "Prediksi", "Tuning Hyperparameter"])

# ===============================
# 1. INFORMASI DATASET
# ===============================
if page == "Informasi Dataset":
    st.title("Informasi Dataset Kepribadian")
    st.write("Dataset ini berisi fitur kepribadian dan label kepribadian target.")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

    st.subheader("Distribusi Label")
    st.bar_chart(df["Personality (Class label)"].value_counts())

# ===============================
# 2. PEMODELAN
# ===============================
elif page == "Pemodelan":
    st.title("Pemodelan Klasifikasi Kepribadian")

    model_choice = st.selectbox("Pilih Model", ["Random Forest", "KNN", "SVM"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    else:
        model = SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y_test, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

# ===============================
# 3. PREDIKSI
# ===============================
elif page == "Prediksi":
    st.title("Prediksi Kepribadian")

    st.write("Masukkan nilai fitur berikut untuk prediksi:")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([inputs])

    model = RandomForestClassifier()
    model.fit(X, y)
    pred = model.predict(input_df)

    st.subheader("Hasil Prediksi")
    st.success(f"Prediksi Kepribadian: {pred[0]}")

    # Unduh hasil
    output = pd.DataFrame(input_df)
    output["Hasil Prediksi"] = pred
    csv = output.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi.csv">Unduh Hasil Prediksi</a>'
    st.markdown(href, unsafe_allow_html=True)

# ===============================
# 4. TUNING HYPERPARAMETER
# ===============================
elif page == "Tuning Hyperparameter":
    st.title("🔧 Tuning Hyperparameter")

    model_select = st.selectbox("Pilih Model", ["Random Forest", "KNN", "SVM"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_select == "Random Forest":
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20]
        }
        model = RandomForestClassifier()
    elif model_select == "KNN":
        param_grid = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
        model = KNeighborsClassifier()
    elif model_select == "SVM":
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
        model = SVC()

    if st.button("🔍 Mulai Tuning"):
        try:
            grid = GridSearchCV(model, param_grid, cv=3)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            st.success("Tuning selesai.")
            st.write("Best Parameters:", grid.best_params_)

            y_pred = best_model.predict(X_test)
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
        except Exception as e:
            st.error("Terjadi error saat tuning. Periksa parameter atau data input.")
            st.exception(e)
