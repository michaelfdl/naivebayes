import pickle
import streamlit as st

nb_model = pickle.load(open('NB_model.sav', 'rb'))

st.title("Sentiment Analysis with Naive Bayes")

st.image("sentiment_polarity.jpg", caption="Sentiment Polarity", use_column_width=True)

user_input = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze Sentiment"):
    prediction = nb_model.predict([user_input])
    st.write(f"Predicted Sentiment: {prediction[0]}")
else:
    st.write("Please enter some text to analyze.")  
    