import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('personality_dataset.csv')
    return df

# Visualisasi data awal
def show_visualizations(df):
    st.subheader("Visualisasi Data")
    st.write("Distribusi Target (Kepribadian):")
    st.bar_chart(df['Kepribadian'].value_counts())

    st.write("Heatmap Korelasi:")
    fig, ax = plt.subplots()
    sns.heatmap(df.drop('Kepribadian', axis=1).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Model training
def train_model(X_train, y_train, model_name):
    if model_name == 'Random Forest':
        model = RandomForestClassifier()
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'SVM':
        model = SVC(probability=True)
    else:
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Visualisasi Confusion Matrix
def show_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues')
    st.pyplot(fig)

# Visualisasi f1-score
def show_f1_plot(report_dict):
    labels = list(report_dict.keys())[:-3]  # remove avg/acc
    f1_scores = [report_dict[label]['f1-score'] for label in labels]

    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=f1_scores, palette='magma', ax=ax)
    ax.set_title("F1-Score per Class")
    st.pyplot(fig)

# Tuning hyperparameter
def tune_model(X, y):
    st.subheader("Tuning Hyperparameter - Random Forest")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10]
    }
    rf = RandomForestClassifier()
    grid = GridSearchCV(rf, param_grid, cv=3)
    grid.fit(X, y)
    st.write("Best Params:", grid.best_params_)
    return grid.best_estimator_

# App Start
st.set_page_config(layout="wide", page_title="Prediksi Kepribadian")

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Tuning Model", "Prediksi", "Anggota Kelompok"])

df = load_data()

if page == "Informasi":
    st.title("Informasi Dataset")
    st.write("Dataset berisi fitur kepribadian dan label Extrovert / Introvert.")
    st.dataframe(df.head())
    show_visualizations(df)

elif page == "Pemodelan Data":
    st.title("📊 Pemodelan Data")
    model_option = st.selectbox("Pilih Model", ['Random Forest', 'KNN', 'SVM'])
    if st.button("Latih Model"):
        X = df.drop("Kepribadian", axis=1)
        y = df["Kepribadian"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = train_model(X_train, y_train, model_option)
        y_pred = model.predict(X_test)

        acc = model.score(X_test, y_test)
        st.metric("Akurasi", f"{acc:.2f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        show_confusion_matrix(y_test, y_pred, labels=model.classes_)

        st.subheader("F1-Score per Class")
        show_f1_plot(report)

elif page == "Tuning Model":
    st.title("⚙️ Tuning Model")
    X = df.drop("Kepribadian", axis=1)
    y = df["Kepribadian"]
    best_model = tune_model(X, y)
    st.success("Model terbaik berhasil didapatkan.")

elif page == "Prediksi":
    st.title("🔍 Prediksi Kepribadian")
    uploaded_file = st.file_uploader("Unggah file CSV untuk prediksi", type=['csv'])
    if uploaded_file is not None:
        pred_df = pd.read_csv(uploaded_file)
        model = RandomForestClassifier().fit(df.drop("Kepribadian", axis=1), df["Kepribadian"])
        pred_result = model.predict(pred_df)
        pred_df['Hasil_Prediksi'] = pred_result
        st.write(pred_df)

        # Unduh hasil
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("Unduh Hasil", data=csv, file_name='hasil_prediksi.csv', mime='text/csv')

elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    st.markdown("""
    - Nama 1 - NIM 1  
    - Nama 2 - NIM 2  
    - Nama 3 - NIM 3  
    - Nama 4 - NIM 4
    """)
