import streamlit as st
import pandas as pd

from stream_sentiment import train_and_evaluate

st.title("Sentiment - Naive Bayes Evaluation")

# Misal data kamu dari CSV
# Pastikan CSV punya kolom "Clean Tweet" dan "Score"
uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Jalankan training+evaluasi
    results = train_and_evaluate(df)

    st.subheader("Confusion Matrix")
    st.dataframe(results["cf_matrix"])

    st.subheader("Classification Report (Table)")
    # report_dict -> jadi tabel
    report_df = pd.DataFrame(results["report_dict"]).transpose()
    st.dataframe(report_df)

    st.subheader("Classification Report (Text)")
    st.code(results["report_text"], language="text")