import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load dataset
file_path = r"C:\Users\hp\Downloads\archive (1)\phishing_site_urls.csv"
df = pd.read_csv(file_path)

# Reduce dataset size for faster training
df = df.sample(n=50000, random_state=42)

# Convert Label column to lowercase and remove spaces
df['Label'] = df['Label'].str.lower().str.strip()

# Convert labels properly
df = df[df['Label'].isin(["good", "bad"])]
df['Label'] = df['Label'].map({"good": 0, "bad": 1})

df.dropna(subset=['Label'], inplace=True)

# Extract features using TF-IDF (Ensure consistent feature size)
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['URL']).toarray()
y = df['Label'].astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
start_time = time.time()
model.fit(X_train, y_train)
print(f"Training Time: {time.time() - start_time} seconds")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\u2705 Model Accuracy: {accuracy:.4f}")

# Save the model and vectorizer
joblib.dump(model, "phishing_url_model.pkl")
joblib.dump(vectorizer, "url_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

# Save the URL model accuracy
url_accuracy = accuracy_score(y_test, y_pred) * 100
with open('url_model_accuracy.txt', 'w') as f:
    f.write(str(url_accuracy))
