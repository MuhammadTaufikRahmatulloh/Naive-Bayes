import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="TA Machine Learning – Naive Bayes",
    layout="centered"
)

# =========================
# Load Dataset (PATH BENAR)
# =========================
df = pd.read_csv("app/Social_Network_Ads.csv")

# =========================
# Prepare Data
# =========================
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Train Model
# =========================
model = GaussianNB()
model.fit(X_train, y_train)

# =========================
# Evaluate Model
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("🧭 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ("Home", "Prediksi")
)

# =========================
# HOME PAGE
# =========================
if menu == "Home":
    st.title("TA Machine Learning – Naive Bayes")

    st.markdown("""
    ### 📌 Overview
    Aplikasi ini merupakan implementasi algoritma **Gaussian Naive Bayes**
    untuk melakukan klasifikasi keputusan pembelian (*Purchased*)
    berdasarkan **Age** dan **Estimated Salary**.

    Dataset yang digunakan adalah **Social Network Ads**.
    """)

    st.markdown("---")

    st.subheader("📊 Preview Dataset")
    st.dataframe(df.head())

    st.subheader("📈 Evaluasi Model")
    st.write(f"Akurasi Model: **{acc:.2f}**")

    # Donut Chart Distribusi Kelas
    st.subheader("🍩 Distribusi Kelas Purchased")
    purchase_counts = df['Purchased'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        purchase_counts,
        labels=["Not Purchased", "Purchased"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    ax.set_title("Distribusi Keputusan Pembelian")
    st.pyplot(fig)

# =========================
# PREDICTION PAGE
# =========================
elif menu == "Prediksi":
    st.title("🔮 Prediksi Data Baru")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=20000,
        max_value=150000,
        value=87000
    )

    if st.button("Prediksi"):
        data = scaler.transform([[age, salary]])
        pred = model.predict(data)
        prob = model.predict_proba(data)[0]

        st.subheader("Hasil Prediksi")
        st.write(
            "Purchased" if pred[0] == 1 else "Not Purchased"
        )

        st.subheader("📊 Probabilitas Prediksi")

        prob_df = pd.DataFrame({
            "Kelas": ["Not Purchased", "Purchased"],
            "Probabilitas": prob
        })

        st.bar_chart(prob_df.set_index("Kelas"))
