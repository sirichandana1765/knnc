import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("KNN Classification - Breast Cancer Dataset")

# Load dataset
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Hyperparameter control
k = st.slider("Select K (Number of Neighbors)", min_value=1, max_value=25, value=5)

test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)

if st.button("Train Model"):

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    st.write("Accuracy:", acc)

    # -----------------------
    # Graph 1: Prediction distribution
    # -----------------------
    st.subheader("Prediction Distribution")

    fig1, ax1 = plt.subplots()
    pd.Series(preds).value_counts().sort_index().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.set_title("Predicted Class Distribution")

    st.pyplot(fig1)

    # -----------------------
    # Graph 2: Actual vs Predicted
    # -----------------------
    st.subheader("Actual vs Predicted (First 30 Samples)")

    fig2, ax2 = plt.subplots()
    ax2.plot(y_test.values[:30], label="Actual")
    ax2.plot(preds[:30], label="Predicted")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Class")
    ax2.legend()
    ax2.set_title("Model Prediction Comparison")

    st.pyplot(fig2)

    # -----------------------
    # Results table
    # -----------------------
    st.subheader("Sample Predictions")
    st.dataframe(pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": preds
    }).head(10))
