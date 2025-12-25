import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Naive Bayes App", layout="centered")

st.title("TA Machine Learning – Naive Bayes")
st.write("Klasifikasi Social Network Ads")

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

st.subheader("Dataset")
st.dataframe(df.head())

# Features & target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Akurasi Model: **{acc:.2f}**")

# User input
st.subheader("Prediksi Data Baru")

age = st.number_input("Age", min_value=18, max_value=60, value=30)
salary = st.number_input("Estimated Salary", min_value=20000, max_value=150000, value=87000)

if st.button("Prediksi"):
    data = scaler.transform([[age, salary]])
    pred = model.predict(data)
    prob = model.predict_proba(data)

    st.write("Hasil Prediksi:", "Purchased" if pred[0] == 1 else "Not Purchased")
    st.write("Probabilitas:", prob)


