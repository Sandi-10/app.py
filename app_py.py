import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from io import BytesIO

# Load dataset
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

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Tuning Model", "Prediksi", "Anggota Kelompok"])

# ---------------- Halaman Informasi ----------------
if page == "Informasi":
    st.title("📘 Informasi Dataset")
    st.write("Dataset ini berisi data kepribadian berdasarkan berbagai fitur")

    st.subheader("Contoh Data")
    st.dataframe(df.head())

    st.subheader("Deskripsi Data")
    st.write(df.describe(include='all'))

    st.subheader("Distribusi Personality")
    fig, ax = plt.subplots()
    sns.countplot(x=target_encoder.inverse_transform(df['Personality']), palette='pastel', ax=ax)
    st.pyplot(fig)

    st.subheader("Korelasi Antar Fitur")
    fig_corr, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig_corr)

# ---------------- Halaman Pemodelan Data ----------------
elif page == "Pemodelan Data":
    st.title("📊 Pemodelan Data")

    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_option = st.selectbox("Pilih Model", ["Random Forest", "KNN", "SVM"])

    if st.button("Latih Model"):
        if model_option == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_option == "KNN":
            model = KNeighborsClassifier()
        elif model_option == "SVM":
            model = SVC(probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)

        st.metric("Akurasi", f"{acc:.2f}")
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_, ax=ax)
        st.pyplot(fig)

        st.subheader("Visualisasi Distribusi Kelas")
        fig_class, ax = plt.subplots()
        sns.countplot(x=target_encoder.inverse_transform(y), ax=ax, palette="pastel")
        st.pyplot(fig_class)

        st.subheader("Heatmap Korelasi Fitur")
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        corr = X.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

        if model_option == "Random Forest":
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'Fitur': X.columns, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False)
            fig_imp, ax = plt.subplots()
            sns.barplot(x='Importance', y='Fitur', data=feat_df, ax=ax)
            st.pyplot(fig_imp)

# ---------------- Halaman Tuning ----------------
elif page == "Tuning Model":
    st.title("🛠️ Tuning Hyperparameter")

    X = df.drop('Personality', axis=1)
    y = df['Personality']

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    grid.fit(X_train, y_train)

    st.subheader("Best Parameters")
    st.write(grid.best_params_)

    y_pred = grid.predict(X_test)
    st.subheader("Akurasi Setelah Tuning")
    st.write(accuracy_score(y_test, y_pred))

# ---------------- Halaman Prediksi ----------------
elif page == "Prediksi":
    st.title("🔮 Prediksi Kepribadian")

    if st.session_state.model is None:
        st.warning("Model belum dilatih.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Personality':
                if df[col].dtype in [np.float64, np.int64]:
                    input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    input_data[col] = st.selectbox(col, sorted(df[col].unique()))

        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                input_df[col] = LabelEncoder().fit(df[col]).transform(input_df[col])

        input_df = input_df[st.session_state.X_columns]
        pred = st.session_state.model.predict(input_df)[0]
        prob = st.session_state.model.predict_proba(input_df)[0]
        label = target_encoder.inverse_transform([pred])[0]

        st.success(f"Prediksi Kepribadian: {label}")
        st.write("Probabilitas:")
        st.bar_chart(pd.Series(prob, index=target_encoder.classes_))

        buffer = BytesIO()
        pd.DataFrame({"Prediction": [label]}).to_csv(buffer, index=False)
        st.download_button("Unduh Hasil", data=buffer.getvalue(), file_name="hasil_prediksi.csv")

# ---------------- Halaman Anggota ----------------
elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    st.markdown("""
    - Diva Auliya Pusparini (2304030041)
    - Paskalia Kanicha Mardian (2304030062)
    - Sandi Krisna Mukti (2304030074)
    - Siti Maisyaroh (2304030079)
    """)
