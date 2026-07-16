import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Classifier",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom attractive & animated CSS
st.markdown("""
    <style>
    /* Gradient Background & Smooth Transitions */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        transition: all 0.5s ease-in-out;
    }
    
    /* Animated Title */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1e3d59;
        font-weight: 700;
        text-align: center;
        animation: fadeInDown 1s ease-out;
        margin-bottom: 10px;
    }
    
    /* Subtitle styling */
    .sub-title {
        text-align: center;
        color: #17b978;
        font-weight: 400;
        margin-bottom: 40px;
    }

    /* Animated Card for Results */
    @keyframes pulseCard {
        0% { transform: scale(0.98); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        animation: pulseCard 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }
    
    .positive-res {
        background-color: #d4edda;
        color: #155724;
        border-left: 8px solid #28a745;
    }
    
    .negative-res {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 8px solid #dc3545;
    }
    </style>
""", unsafe_allowed_html=True)

# Title section
st.markdown("<h1 class='main-title'>🔮 AI Text Sentiment Classifier</h1>", unsafe_allowed_html=True)
st.markdown("<p class='sub-title'>Enter your text below to predict whether the sentiment is Positive or Negative.</p>", unsafe_allowed_html=True)

# Cache model and vectorizer loading for performance
@st.cache_resource
def load_assets():
    with open("vector.pkl", "rb") as v_file:
        vectorizer = pickle.load(v_file)
    with open("model (10).pkl", "rb") as m_file:
        model = pickle.load(m_file)
    return vectorizer, model

try:
    vectorizer, model = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure 'vector.pkl' and 'model (10).pkl' are in the same folder.")
    st.stop()

# Input UI Layout
with st.container():
    user_text = st.text_area("✍️ Type your text message here:", height=150, placeholder="Type something like 'I absolutely love this production! The scenery was spectacular...'")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # A styled submit button
        submit_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

# Processing and Prediction
if submit_btn:
    if user_text.strip() == "":
        st.warning("Please enter some text before analyzing!")
    else:
        with st.spinner("Analyzing text dynamics... Please wait..."):
            # 1. Vectorize input text
            vectorized_input = vectorizer.transform([user_text])
            
            # 2. Predict using Logistic Regression
            prediction = model.predict(vectorized_input)[0]
            
            # 3. Get prediction probabilities (optional, for metrics visualization)
            try:
                probabilities = model.predict_proba(vectorized_input)[0]
                class_labels = model.classes_
                prob_dict = dict(zip(class_labels, probabilities))
            except AttributeError:
                prob_dict = None

        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allowed_html=True)
        
        # Display Animated Results
        if prediction == "positive":
            st.markdown(
                f"<div class='result-card positive-res'>✨ Predicted Sentiment: POSITIVE ✨</div>", 
                unsafe_allowed_html=True
            )
            st.balloons()  # Adds interactive animation for positive results
        else:
            st.markdown(
                f"<div class='result-card negative-res'>⚠️ Predicted Sentiment: NEGATIVE ⚠️</div>", 
                unsafe_allowed_html=True
            )
            st.snow()  # Soft downward animation effect

        # Display Confidence level if available
        if prob_dict:
            st.write("")
            st.subheader("📊 Confidence Breakdown")
            confidence = prob_dict[prediction] * 100
            st.progress(int(confidence))
            st.write(f"Model Confidence: **{confidence:.2f}%**")
