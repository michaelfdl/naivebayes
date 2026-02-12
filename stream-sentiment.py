import pickle
import streamlit as st

nb_model = pickle.load(open('NB_model.sav', 'rb'))

st.title("Sentiment Analysis with Naive Bayes")

st.image("sentiment_polarity.jpg", caption="Sentiment Polarity")
    