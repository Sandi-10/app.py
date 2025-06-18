import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Load dataset
url = 'https://raw.githubusercontent.com/Sandi-10/Personality/main/personality_dataset.csv'
df = pd.read_csv(url)

# Encode target
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

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi", "Pemodelan Data", "Tuning Model", "Prediksi", "Anggota Kelompok"])

# Informasi
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

    st.subheader("📦 Boxplot Setiap Fitur Numerik")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col != 'Personality':
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Personality', y=col, ax=ax)
            ax.set_title(f"Distribusi {col} berdasarkan Personality")
            ax.set_xticklabels(target_encoder.inverse_transform(sorted(df['Personality'].unique())))
            st.pyplot(fig)

    st.subheader("🎯 Proporsi Tipe Kepribadian")
    personality_counts = df['Personality'].value_counts()
    labels = target_encoder.inverse_transform(personality_counts.index)
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(personality_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')
    st.pyplot(fig_pie)

    st.subheader("📊 Histogram Setiap Fitur")
    selected_num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in selected_num_cols:
        if col != 'Personality':
            fig, ax = plt.subplots()
            for label in df['Personality'].unique():
                subset = df[df['Personality'] == label]
                ax.hist(subset[col], alpha=0.5, label=target_encoder.inverse_transform([label])[0])
            ax.set_title(f'Distribusi {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frekuensi')
            ax.legend()
            st.pyplot(fig)

    st.subheader("📚 Rata-rata Skor Fitur Berdasarkan Tipe Kepribadian")
    mean_features = df.groupby('Personality').mean(numeric_only=True)
    mean_features.index = target_encoder.inverse_transform(mean_features.index)
    st.dataframe(mean_features.style.format("{:.2f}"))

    for col in mean_features.columns:
        fig, ax = plt.subplots()
        sns.barplot(x=mean_features.index, y=mean_features[col], ax=ax, palette='coolwarm')
        ax.set_title(f"Rata-rata {col} per Kepribadian")
        st.pyplot(fig)

# Halaman lain tetap dilanjutkan seperti sebelumnya (Pemodelan Data, Tuning, Prediksi, Anggota)
# ...
