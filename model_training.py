import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess_data
import time

start = time.time()

# Load your old dataset
old_df = pd.read_csv("Phishing_Email.csv", usecols=["Email Text", "Email Type"], low_memory=False)

# Load the new Zenodo phishing dataset
new_df = pd.read_csv("Phishing_validation_emails.csv")

# Drop unnecessary columns if present
old_df = old_df.drop(columns=['Serial No.'], errors='ignore')
old_df.columns = ['Email Text', 'Email Type']
new_df.columns = ['Email Text', 'Email Type']

# Clean any accidental whitespace or formatting issues
old_df["Email Type"] = old_df["Email Type"].astype(str).str.strip()
new_df["Email Type"] = new_df["Email Type"].astype(str).str.strip()

# Merge datasets and shuffle
combined_df = pd.concat([old_df, new_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Only keep rows with valid labels
valid_labels = ['Phishing Email', 'Safe Email']
cleaned_df = combined_df[combined_df['Email Type'].isin(valid_labels)].copy()

print(f"✅ Cleaned dataset shape: {cleaned_df.shape}")

# Load dataset
X, y, vectorizer = load_and_preprocess_data(cleaned_df)

print("Total samples after merge:", combined_df.shape[0])
#print("Label distribution:\n", combined_df["Email Type"].value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "phishing_email_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n💾 Model and vectorizer saved successfully.")

print(f"\n⏱️ Time taken: {round(time.time() - start, 2)} seconds")

# Save the email model accuracy
email_accuracy = accuracy_score(y_test, y_pred) * 100
with open('email_model_accuracy.txt', 'w') as f:
    f.write(str(email_accuracy))
