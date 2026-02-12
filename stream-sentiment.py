import pickle
import streamlit as st
#from naiveBayes import cf_matrix

nb_model = pickle.load(open('NB_model.sav', 'rb'))

st.title("Sentiment Analysis with Naive Bayes", text_alignment="center",width=800)
st.header("sentiment polarity prediction")
st.image("sentiment_polarity.jpg", caption="Sentiment Polarity")

st.table(cf_matrix)