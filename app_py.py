import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from io import BytesIO

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

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Tuning Model", "Prediksi", "Anggota Kelompok"])

# -------------------- Halaman Informasi --------------------
if page == "Informasi":
    st.title("📘 Informasi Dataset")
    st.dataframe(df.head())
    st.subheader("📊 Deskripsi Kolom")
    st.write(df.describe(include='all'))

    st.subheader("📈 Histogram Fitur Numerik")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribusi {col}")
        st.pyplot(fig)

    st.subheader("📦 Boxplot Setiap Fitur Numerik")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, y=col, ax=ax)
        ax.set_title(f"Boxplot {col}")
        st.pyplot(fig)

    st.subheader("🎯 Distribusi Target (Pie Chart)")
    personality_counts = df['Personality'].value_counts()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(personality_counts, labels=target_encoder.inverse_transform(personality_counts.index), autopct='%1.1f%%', startangle=90)
    st.pyplot(fig_pie)

    st.subheader("📉 Korelasi antar Fitur")
    corr = df.corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("📊 Rata-rata Fitur per Personality")
    means = df.groupby('Personality').mean(numeric_only=True)
    for col in means.columns:
        fig_bar, ax_bar = plt.subplots()
        means[col].plot(kind='bar', ax=ax_bar)
        ax_bar.set_title(f"Rata-rata {col} per Personality")
        ax_bar.set_xticklabels(target_encoder.inverse_transform(means.index))
        st.pyplot(fig_bar)

# -------------------- Halaman Pemodelan --------------------
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

        st.metric(label="Akurasi", value=f"{acc:.2f}")
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig_cm)

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances}).sort_values(by='Pentingnya', ascending=False)
            fig_imp, ax2 = plt.subplots()
            sns.barplot(x='Pentingnya', y='Fitur', data=imp_df, palette='viridis', ax=ax2)
            st.pyplot(fig_imp)

# -------------------- Halaman Tuning Model --------------------
elif page == "Tuning Model":
    st.title("🔧 Tuning Hyperparameter")
    model_choice = st.selectbox("Pilih Model untuk Tuning", ["Random Forest", "SVM", "KNN"])

    df_model = df.copy()
    X = df_model.drop('Personality', axis=1)
    y = df_model['Personality']

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "Random Forest":
        param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    elif model_choice == "KNN":
        param_grid = {'n_neighbors': [3, 5, 7]}
        model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
    else:
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        model = GridSearchCV(SVC(probability=True), param_grid, cv=3)

    if st.button("🔍 Mulai Tuning"):
        model.fit(X_train, y_train)
        st.write("Best Params:", model.best_params_)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write("Akurasi:", acc)

# -------------------- Halaman Prediksi --------------------
elif page == "Prediksi":
    st.title("🔮 Prediksi Kepribadian")

    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan buka halaman 'Pemodelan Data'.")
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
            predicted_label = target_encoder.inverse_transform([prediction])[0]
            st.success(f"✅ Tipe Kepribadian yang Diprediksi: *{predicted_label}*")

            st.subheader("📋 Input Anda")
            st.dataframe(input_df)

            st.download_button(
                label="📥 Unduh Hasil Prediksi",
                data=BytesIO(input_df.to_csv(index=False).encode()),
                file_name='hasil_prediksi.csv',
                mime='text/csv'
            )

# -------------------- Halaman Anggota Kelompok --------------------
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
