import streamlit as st
import requests

st.title("AI-Based Exam Anxiety Detector")

text = st.text_area(
    "Enter your thoughts before exam:",
    placeholder="Example: I am worried about my exam and can't focus properly..."
)

if st.button("Analyze Anxiety"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    
    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": text}
        )

        if response.status_code == 200:
            result = response.json()
            level = result["anxiety_level"]

            st.success(f"Predicted Anxiety Level: {level}")