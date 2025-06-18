import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Fungsi untuk mengonversi gambar ke base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Tambahkan background gambar ke seluruh halaman
bg_image = get_base64("a14f21d8-501c-4e9f-86d7-79e649c615c8.jpg")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# Encode target
target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

# Inisialisasi session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

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

    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Pilih Model", ["Random Forest", "KNN", "SVM"])

    if st.button("🚀 Latih Model"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        else:
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
        st.warning("Model belum dilatih.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Personality':
                if df[col].dtype in [np.float64, np.int64]:
                    val = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    val = st.selectbox(col, sorted(df[col].dropna().unique()))
                input_data[col] = val

        input_df = pd.DataFrame([input_data])
        for col in input_df.select_dtypes(include='object').columns:
            input_df[col] = LabelEncoder().fit(df[col]).transform(input_df[col])

        input_df = input_df[st.session_state.X_columns]

        if st.button("Prediksi"):
            model = st.session_state.model
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            predicted_label = target_encoder.inverse_transform([prediction])[0]
            st.session_state.prediction_result = pd.DataFrame({
                'Predicted Personality': [predicted_label],
                **{f"Prob_{label}": [p] for label, p in zip(target_encoder.classes_, prob)}
            })

            st.success(f"✅ Tipe Kepribadian: *{predicted_label}*")
            st.subheader("📈 Probabilitas")
            st.bar_chart(pd.Series(prob, index=target_encoder.classes_))

        if st.session_state.prediction_result is not None:
            st.subheader("📥 Unduh Hasil Prediksi")
            csv = st.session_state.prediction_result.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi.csv">📥 Klik untuk unduh CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

# -------------------- Halaman Tuning Model --------------------
elif page == "Tuning Model":
    st.title("🎯 Tuning Hyperparameter")

    X = df.drop('Personality', axis=1)
    y = df['Personality']

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_name = st.selectbox("Pilih Model untuk Tuning", ["Random Forest", "KNN", "SVM"])

    if model_name == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }
        model = RandomForestClassifier(random_state=42)
    elif model_name == "KNN":
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
        model = KNeighborsClassifier()
    else:
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        model = SVC(probability=True)

    if st.button("Mulai Grid Search"):
        with st.spinner("⏳ Mencari parameter terbaik..."):
            grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            st.success("✅ Selesai!")
            st.write("Best Parameters:", grid.best_params_)
            st.write("Best Score:", grid.best_score_)
            st.session_state.model = grid.best_estimator_

# -------------------- Halaman Anggota --------------------
elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("a14f21d8-501c-4e9f-86d7-79e649c615c8.jpg", width=180)
    with col2:
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
