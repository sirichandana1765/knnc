import streamlit as st
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")
st.title("KNN Classification")

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

st.dataframe(df.head())

if st.button("Train Model"):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, preds))

    st.bar_chart(pd.DataFrame({"Predicted": preds}))
