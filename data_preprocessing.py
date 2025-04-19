import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """Cleans email content."""
    if pd.isnull(text):  # Handle missing values
        return ""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

def load_and_preprocess_data(df):
    """Loads CSV file and preprocesses email text."""
    df["cleaned_email"] = df["Email Text"].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["cleaned_email"]).toarray()
    df["Email Type"] = df["Email Type"].astype(str).str.strip()
    df["Email Type"] = df["Email Type"].map({"Phishing Email": 1, "Safe Email": 0})

    # Optional sanity check
    if df["Email Type"].isnull().any():
        raise ValueError("Found unmapped labels! Check for typos or unknown types.")

    y = df["Email Type"]

    return X, y, vectorizer
