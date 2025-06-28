import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("spam_classifier_model.pkl")

# Title
st.title("📧 Spam Email Classifier")
st.markdown("Enter the values for a new email below to check whether it's **Spam** or **Not Spam**.")

# Input form
st.header("✍️ Email Features")

# Create 57 input fields (you can customize this)
inputs = []
for i in range(57):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

# Predict button
if st.button("🚀 Predict"):
    user_input = np.array(inputs).reshape(1, -1)
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.error("⚠️ This is **SPAM**!")
    else:
        st.success("✅ This is **NOT SPAM**.")
