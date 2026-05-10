"""
Train and save the resume classifier model
Run this file ONCE before starting the Streamlit app
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

print("=" * 60)
print("TRAINING RESUME CLASSIFIER MODEL")
print("=" * 60)

# Load preprocessed data
print("\n1. Loading data...")
df = pd.read_csv('../outputs/preprocessed_resumes.csv')
print(f"   Loaded {len(df)} resumes")
print(f"   Categories: {df['Category'].nunique()}")

# Prepare features and labels
print("\n2. Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['processed_resume'])
y = df['Category']
print(f"   Feature matrix: {X.shape}")

# Train model
print("\n3. Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)
print("   Training complete!")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save model and vectorizer
print("\n4. Saving model files...")
joblib.dump(model, 'models/classifier.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("   ✓ Saved to 'models/' folder")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nNow run: streamlit run app.py")