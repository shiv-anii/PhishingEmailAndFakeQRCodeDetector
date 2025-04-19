import streamlit as st
import joblib
import cv2
import numpy as np
import re
from data_preprocessing import preprocess_text

# Load trained models and vectorizers
email_model = joblib.load("phishing_email_model.pkl")
email_vectorizer = joblib.load("vectorizer.pkl")
url_model = joblib.load("phishing_url_model.pkl")
url_vectorizer = joblib.load("url_vectorizer.pkl")

# Load accuracies
try:
    with open('email_model_accuracy.txt', 'r') as f:
        email_model_accuracy = float(f.read())
except:
    email_model_accuracy = 0.0  # fallback if file not found

try:
    with open('url_model_accuracy.txt', 'r') as f:
        url_model_accuracy = float(f.read())
except:
    url_model_accuracy = 0.0  # fallback if file not found

# Combined model accuracy (for "Both" option)
combined_model_accuracy = (email_model_accuracy + url_model_accuracy) / 2

# Function to predict email category
def predict_email_category(email_text):
    processed_email = email_vectorizer.transform([preprocess_text(email_text)]).toarray()
    prediction = email_model.predict(processed_email)[0]
    confidence = max(email_model.predict_proba(processed_email)[0]) * 100
    return ("Phishing" if prediction == 1 else "Legitimate"), confidence

# Extract QR Code Data using OpenCV
def extract_qr_code(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    qr_detector = cv2.QRCodeDetector()
    data, _, _ = qr_detector.detectAndDecode(image)
    return data if data else "No QR code detected."

# Extract URL Features
def extract_url_features(url):
    features = url_vectorizer.transform([url]).toarray()
    return features

# Function to predict URL category
def predict_url_category(url):
    features = extract_url_features(url)
    prediction = url_model.predict(features)[0]
    confidence = max(url_model.predict_proba(features)[0]) * 100
    return ("Phishing" if prediction == 1 else "Legitimate"), confidence

# Streamlit UI
st.title("Phishing Email and QR Code Detector")

# User selects the input type
option = st.radio("Select Input Type:", ["Only Text", "Only QR Code", "Both"])

if option == "Only Text":
    email_content = st.text_area("Enter email content:")
    if st.button("Analyze Email"):
        result, confidence = predict_email_category(email_content)
        st.write(f"Result: {result} (Confidence: {confidence:.2f}%)")
        st.write(f"\n✅ Model Accuracy: {email_model_accuracy:.2f}%")

elif option == "Only QR Code":
    qr_file = st.file_uploader("Upload a QR code image:", type=["jpg", "jpeg", "png", "pdf"])
    if qr_file:
        qr_data = extract_qr_code(qr_file)
        st.write(f"QR Code Data: {qr_data}")
        if qr_data and qr_data != "No QR code detected.":
            result, confidence = predict_url_category(qr_data)
            st.write(f"QR Code Classification: {result} (Confidence: {confidence:.2f}%)")
            st.write(f"\n✅ Model Accuracy: {url_model_accuracy:.2f}%")
        else:
            st.write("No valid QR code found.")

elif option == "Both":
    email_file = st.file_uploader("Upload email file (text/eml/pdf):", type=["txt", "eml", "pdf"])
    qr_file = st.file_uploader("Upload a QR code image:", type=["jpg", "jpeg", "png", "pdf"])
    if email_file and qr_file:
        email_content = email_file.read().decode("utf-8")
        email_result, email_confidence = predict_email_category(email_content)
        qr_data = extract_qr_code(qr_file)
        st.write(f"Email Result: {email_result} (Confidence: {email_confidence:.2f}%)")
        if qr_data and qr_data != "No QR code detected.":
            qr_result, qr_confidence = predict_url_category(qr_data)
            st.write(f"QR Code Classification: {qr_result} (Confidence: {qr_confidence:.2f}%)")
            st.write(f"\n✅ Model Accuracy: {combined_model_accuracy:.2f}%")

