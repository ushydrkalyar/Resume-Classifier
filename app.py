"""
Resume Analyzer - Professional Resume Classification System
A Streamlit application to analyze and classify resumes by job category
"""

import streamlit as st
import pandas as pd
import joblib
import PyPDF2
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
from datetime import datetime

# Download NLTK data (only once)
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #00c6fb 0%, #005bea 100%);
        color: white;
        padding: 0.25rem 0.5rem;
        text-align: right;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None

# Load model and vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/classifier.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run 'python model_trainer.py' first.")
        return None, None

model, vectorizer = load_models()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def preprocess_text(text):
    """Apply same preprocessing as training"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove non-alphabetic and stop words
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def analyze_resume(text):
    """Predict job category from resume text"""
    if model is None or vectorizer is None:
        return None, None
    
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Transform using saved vectorizer
    features = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Get probabilities
    probabilities = model.predict_proba(features)[0]
    
    # Get all classes
    classes = model.classes_
    
    # Create probability dictionary
    prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}
    
    # Sort by probability
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    return prediction, sorted_probs

# Main Header
st.markdown("""
<div class="main-header">
    <h1>📄 Professional Resume Analyzer</h1>
    <p>AI-Powered Resume Classification System</p>
</div>
""", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Resume")
    st.markdown("Upload a PDF resume to analyze and classify")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload resume in PDF format"
    )
    
    if uploaded_file is not None:
        with st.spinner("📖 Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            
        if resume_text:
            st.success("✅ Resume uploaded successfully!")
            
            # Show file info
            st.markdown("**File Details:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("File Name", uploaded_file.name[:30] + "...")
            with col_b:
                st.metric("Text Length", f"{len(resume_text):,} characters")
            
            # Preview
            with st.expander("📄 Resume Preview"):
                st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
            
            # Analyze button
            if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
                with st.spinner("🧠 Analyzing resume content..."):
                    prediction, probabilities = analyze_resume(resume_text)
                
                if prediction:
                    st.session_state.prediction = prediction
                    st.session_state.probabilities = probabilities
                    st.session_state.processed = True

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if st.session_state.processed and st.session_state.prediction:
        # Display prediction
        st.markdown("""
        <div class="result-card">
            <h4 style="color: #667eea;">🎯 Predicted Job Category</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="prediction">
            {st.session_state.prediction}
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence scores
        st.markdown("### 📈 Confidence Scores")
        
        for category, prob in st.session_state.probabilities[:5]:
            percentage = int(prob * 100)
            st.markdown(f"""
            <div>
                <strong>{category}</strong>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {percentage}%;">
                        {percentage}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 💡 Insights")
        
        top_category = st.session_state.probabilities[0][0]
        top_confidence = st.session_state.probabilities[0][1] * 100
        
        if top_confidence > 80:
            st.success(f"✅ High confidence match for **{top_category}**")
        elif top_confidence > 60:
            st.warning(f"⚠️ Medium confidence match for **{top_category}**")
        else:
            st.error(f"❌ Low confidence - resume may need more details")
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        
        recommendations = {
            "Data Scientist": "Highlight statistics, machine learning algorithms, and data visualization skills",
            "Software Engineer": "Emphasize programming languages, frameworks, and project experience",
            "Frontend Developer": "Showcase React/Angular/Vue, responsive design, and UI/UX skills",
            "Backend Developer": "Focus on databases, APIs, server architecture, and system design",
            "Full Stack Developer": "Mention both frontend and backend technologies",
            "Machine Learning Engineer": "Include model deployment, MLOps, and production experience",
            "Cloud Engineer": "Highlight AWS/Azure/GCP, containerization, and DevOps tools"
        }
        
        if st.session_state.prediction in recommendations:
            st.info(f"💡 {recommendations[st.session_state.prediction]}")
        
    elif st.session_state.processed and not st.session_state.prediction:
        st.error("❌ Could not analyze resume. Please try again.")
    else:
        st.info("👈 Upload a PDF resume and click 'Analyze Resume' to see results")

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This AI-powered resume analyzer uses **Machine Learning** to classify 
    resumes into job categories based on their content.
    """)
    
    st.markdown("---")
    st.markdown("### 🏆 Supported Categories")
    
    if model is not None:
        categories = model.classes_
        for cat in categories:
            st.markdown(f"• {cat}")
    
    st.markdown("---")
    st.markdown("### 📊 How It Works")
    st.markdown("""
    1. **Extract** text from uploaded PDF
    2. **Preprocess** (tokenization, stop word removal, lemmatization)
    3. **Transform** using TF-IDF vectorizer
    4. **Predict** category using Logistic Regression
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    - Streamlit (UI)
    - scikit-learn (ML)
    - NLTK (Text Processing)
    - PyPDF2 (PDF Extraction)
    """)
    
    st.markdown("---")
    st.markdown(f"**Version:** 1.0 | **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>Professional Resume Analyzer | Powered by Machine Learning | © 2026</p>
</div>
""", unsafe_allow_html=True)