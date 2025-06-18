# Streamlit App Lengkap untuk Prediksi Kepribadian dengan Pemilihan Model
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load data
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

target_encoder = LabelEncoder()
df['Personality'] = target_encoder.fit_transform(df['Personality'])

if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Informasi", "Pemodelan Data", "Prediksi", "Anggota Kelompok"])

if page == "Informasi":
    st.title("📘 Informasi Dataset")
    st.dataframe(df.head())

    st.subheader("Distribusi Target")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Personality', ax=ax1)
    ax1.set_xticklabels(target_encoder.inverse_transform(sorted(df['Personality'].unique())))
    st.pyplot(fig1)

    st.subheader("Korelasi antar Fitur")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Boxplot Fitur")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col != 'Personality':
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Personality', y=col, ax=ax)
            ax.set_xticklabels(target_encoder.inverse_transform(sorted(df['Personality'].unique())))
            st.pyplot(fig)

elif page == "Pemodelan Data":
    st.title("🧠 Pemodelan Data")

    X = df.drop('Personality', axis=1)
    y = df['Personality']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    test_size = st.slider("Ukuran Data Uji", 0.1, 0.5, 0.2)
    model_choice = st.selectbox("Pilih Model", ["Random Forest", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Logistic Regression"])

    params = {}
    if model_choice == "Random Forest":
        params['n_estimators'] = st.number_input("Jumlah Pohon", 10, 200, 100)
    elif model_choice == "K-Nearest Neighbors":
        params['n_neighbors'] = st.number_input("Jumlah Tetangga", 1, 20, 5)
    elif model_choice == "Support Vector Machine":
        params['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if st.button("Latih Model"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=42)
        elif model_choice == "K-Nearest Neighbors":
            model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        elif model_choice == "Support Vector Machine":
            model = SVC(kernel=params['kernel'], probability=True)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.X_columns = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)).transpose().style.format("{:.2f}"))

        fig_cm, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_, ax=ax)
        st.pyplot(fig_cm)

        if hasattr(model, "feature_importances_"):
            fig_imp, ax = plt.subplots()
            imp_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': model.feature_importances_})
            sns.barplot(data=imp_df.sort_values('Pentingnya', ascending=False), x='Pentingnya', y='Fitur', ax=ax)
            st.pyplot(fig_imp)

        if len(target_encoder.classes_) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title("ROC Curve")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
            st.pyplot(fig_roc)

elif page == "Prediksi":
    st.title("🔮 Prediksi Kepribadian")
    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan latih model terlebih dahulu.")
    else:
        input_data = {}
        for col in df.columns:
            if col != 'Personality':
                if df[col].dtype in [np.float64, np.int64]:
                    input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                else:
                    input_data[col] = st.selectbox(col, sorted(df[col].dropna().unique()))

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

            st.success(f"Prediksi Kepribadian: {label}")
            st.bar_chart(pd.Series(prob, index=target_encoder.classes_))

elif page == "Anggota Kelompok":
    st.title("👥 Anggota Kelompok")
    st.markdown("""
    - Diva Auliya Pusparini (2304030041)
    - Paskalia Kanicha Mardian (2304030062)
    - Sandi Krisna Mukti (2304030074)
    - Siti Maisyaroh (2304030079)
    """)
