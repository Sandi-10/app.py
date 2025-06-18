import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from PIL import Image

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# ==================== Fungsi Tambahan ====================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def encode_input(df_input, df_ref):
    df_encoded = df_input.copy()
    for col in df_input.columns:
        if df_input[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(df_ref[col])
            df_encoded[col] = le.transform(df_input[col])
    return df_encoded

# ==================== Load & Persiapan Data ====================
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)
df_raw = df.copy()

# Encode target
target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

# Encode fitur kategorikal
X = df.drop('Personality', axis=1)
y = df['Personality']
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================== Streamlit Setup ====================
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
    """, unsafe_allow_html=True
)

# ==================== Navigasi Halaman ====================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Tuning Model", "Prediksi", "Anggota Kelompok"])

# ==================== Halaman Informasi ====================
if page == "Informasi":
    st.title("📘 Informasi Dataset")
    st.write("Dataset ini berisi data kepribadian berdasarkan berbagai aspek.")
    st.dataframe(df_raw.head())
    st.write(df_raw.describe(include='all'))

    st.subheader("Distribusi Personality")
    fig, ax = plt.subplots()
    sns.countplot(x=target_encoder.inverse_transform(df['Personality']), ax=ax)
    ax.set_xlabel("Personality")
    st.pyplot(fig)

# ==================== Halaman Pemodelan ====================
elif page == "Pemodelan Data":
    st.title("📊 Pemodelan Data")
    model_option = st.selectbox("Pilih Model", ["Random Forest", "SVM", "KNN"])

    if st.button("Latih Model"):
        if model_option == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_option == "SVM":
            model = SVC(probability=True)
        elif model_option == "KNN":
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Akurasi", f"{acc:.2f}")
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

        st.subheader("Confusion Matrix")
        fig_cm, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_encoder.classes_,
                    yticklabels=target_encoder.classes_, ax=ax)
        st.pyplot(fig_cm)

        # Simpan model ke sesi
        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()

# ==================== Halaman Tuning Model ====================
elif page == "Tuning Model":
    st.title("🎯 Tuning Hyperparameter - Random Forest")
    if st.button("Lakukan Tuning"):
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        rs = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=5, cv=3, random_state=42)
        rs.fit(X_train, y_train)
        st.write("Best Params:", rs.best_params_)
        y_pred = rs.predict(X_test)
        st.metric("Akurasi Tuning", f"{accuracy_score(y_test, y_pred):.2f}")
        st.session_state.model = rs.best_estimator_

# ==================== Halaman Prediksi ====================
elif page == "Prediksi":
    st.title("🔮 Prediksi Kepribadian")
    if 'model' not in st.session_state:
        st.warning("Model belum dilatih. Silakan latih di halaman Pemodelan.")
    else:
        input_data = {}
        for col in df_raw.columns:
            if col != 'Personality':
                if df_raw[col].dtype == object:
                    input_data[col] = st.selectbox(col, df_raw[col].unique())
                else:
                    input_data[col] = st.slider(col, int(df_raw[col].min()), int(df_raw[col].max()), int(df_raw[col].mean()))
        input_df = pd.DataFrame([input_data])
        input_encoded = encode_input(input_df, df_raw)

        if st.button("Prediksi"):
            model = st.session_state.model
            input_encoded = input_encoded[st.session_state.X_columns]
            pred = model.predict(input_encoded)[0]
            prob = model.predict_proba(input_encoded)[0]
            result = target_encoder.inverse_transform([pred])[0]
            st.success(f"Tipe Kepribadian yang Diprediksi: {result}")

            st.subheader("Probabilitas:")
            prob_df = pd.Series(prob, index=target_encoder.classes_)
            st.bar_chart(prob_df)

            # Unduh hasil
            hasil_df = pd.DataFrame({**input_data, "Prediksi": result}, index=[0])
            csv = hasil_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Hasil", csv, file_name="hasil_prediksi.csv", mime='text/csv')

# ==================== Halaman Anggota ====================
elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("a14f21d8-501c-4e9f-86d7-79e649c615c8.jpg", width=180)
    with col2:
        st.markdown("""
        ### 👩‍🏫 *Diva Auliya Pusparini*  
        🆔 NIM: 2304030041  

        ### 👩‍🎓 *Paskalia Kanicha Mardian*  
        🆔 NIM: 2304030062  

        ### 👨‍💻 *Sandi Krisna Mukti*  
        🆔 NIM: 2304030074  

        ### 👩‍⚕ *Siti Maisyaroh*  
        🆔 NIM: 2304030079
        """)
