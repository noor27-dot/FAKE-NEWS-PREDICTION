
import streamlit as st
import pickle
import re

# Load model & vectorizer
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction function
def predict_fake_news(text):
    text_clean = clean_text(text)
    vector = vectorizer.transform([text_clean])
    pred = model.predict(vector)[0]
    return "Real News" if pred==1 else "Fake News"

# Streamlit UI
st.title("Fake News Detection")
st.write("Enter a news article below to check if it is Fake or Real.")
user_input = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_fake_news(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text!")
