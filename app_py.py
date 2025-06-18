import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from io import BytesIO

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Fungsi simpan background base64 (tidak dipakai karena error file)
# def get_base64(file_path):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# Load data
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# Encode target
target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

# Inisialisasi session state
for key in ['model', 'X_columns', 'X_test', 'y_test', 'last_prediction']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Prediksi", "Tuning Model", "Anggota Kelompok"])

# -------------------- Halaman Informasi --------------------
if page == "Informasi":
    st.title("📘 Informasi Dataset")
    st.write("Dataset ini berisi data kepribadian berdasarkan berbagai aspek.")

    st.subheader("🔍 Contoh Data")
    st.dataframe(df.head())

    st.subheader("📊 Deskripsi Kolom")
    st.write(df.describe(include='all'))

    st.subheader("🧠 Distribusi Target (Personality Type)")
    fig_dist, ax_dist = plt.subplots()
    sns.countplot(data=df, x='Personality', ax=ax_dist)
    ax_dist.set_xticklabels(target_encoder.inverse_transform(sorted(df['Personality'].unique())))
    st.pyplot(fig_dist)

    st.subheader("📉 Korelasi antar Fitur")
    fig_corr, ax_corr = plt.subplots()
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# -------------------- Halaman Pemodelan --------------------
elif page == "Pemodelan Data":
    st.title("📊 Pemodelan Data")

    model_choice = st.selectbox("Pilih Model", ["RandomForest", "KNN", "SVM"])
    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    # Encode fitur kategorikal
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("🚀 Latih Model"):
        if model_choice == "RandomForest":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        elif model_choice == "SVM":
            model = SVC(probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.subheader("🎯 Akurasi Model")
        st.metric(label="Akurasi", value=f"{acc:.2f}")

        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_encoder.classes_,
                    yticklabels=target_encoder.classes_, ax=ax)
        st.pyplot(fig_cm)

# -------------------- Halaman Prediksi --------------------
elif page == "Prediksi":
    st.title("🔮 Prediksi Kepribadian")

    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan buka halaman 'Pemodelan Data' dan klik 'Latih Model'.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Personality':
                if df[col].dtype in [np.float64, np.int64]:
                    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    val = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
                input_data[col] = val

        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi"):
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(df[col])
                    input_df[col] = le.transform(input_df[col])

            input_df = input_df[st.session_state.X_columns]
            prediction = st.session_state.model.predict(input_df)[0]
            prob = st.session_state.model.predict_proba(input_df)[0]
            predicted_label = target_encoder.inverse_transform([prediction])[0]

            st.success(f"✅ Tipe Kepribadian: *{predicted_label}*")
            st.session_state.last_prediction = pd.DataFrame({
                'Predicted Personality': [predicted_label],
                **input_data
            })

            st.subheader("📈 Probabilitas Prediksi")
            prob_df = pd.Series(prob, index=target_encoder.classes_)
            st.bar_chart(prob_df)

            csv = st.session_state.last_prediction.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Unduh Hasil Prediksi", data=csv, file_name='hasil_prediksi.csv', mime='text/csv')

# -------------------- Halaman Tuning Model --------------------
elif page == "Tuning Model":
    st.title("🛠️ Tuning Hyperparameter")

    model_option = st.selectbox("Pilih Model", ["RandomForest", "KNN", "SVM"])
    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    for col in X.select_dtypes("object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_option == "RandomForest":
        params = {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20]
        }
        model = RandomForestClassifier(random_state=42)
    elif model_option == "KNN":
        params = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
        model = KNeighborsClassifier()
    elif model_option == "SVM":
        params = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
        model = SVC(probability=True)

    if st.button("🔍 Mulai Tuning"):
        grid = GridSearchCV(model, params, cv=3)
        grid.fit(X_train, y_train)

        st.success(f"✅ Best Parameters: {grid.best_params_}")
        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Akurasi Terbaik", f"{acc:.2f}")

# -------------------- Halaman Anggota --------------------
elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    st.markdown("""
    ### 👩‍🏫 Diva Auliya Pusparini  
    🆔 NIM: 2304030041

    ### 👩‍🎓 Paskalia Kanicha Mardian  
    🆔 NIM: 2304030062

    ### 👨‍💻 Sandi Krisna Mukti  
    🆔 NIM: 2304030074

    ### 👩‍⚕ Siti Maisyaroh  
    🆔 NIM: 2304030079
    """)
