# 📄 Resume Classifier

AI-powered resume classification system that predicts job categories from PDF resumes.

## 🎯 What It Does

Upload a resume → Get predicted job role with confidence score.

**Supported Categories:** Data Scientist, Frontend Developer, Backend Developer, Full Stack Developer, ML Engineer, Cloud Engineer, Mobile Developer, Python Developer

## 🛠️ Tech Stack

- Streamlit (UI)
- Logistic Regression (ML Model)
- TF-IDF (Feature Extraction)
- NLTK (Text Processing)
- PyPDF2 (PDF Extraction)

## 🚀 Quick Start

# 1. Install
pip install -r requirements.txt

# 2. Train model (first time only)
python model_trainer.py

# 3. Run app
streamlit run app.py
