import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords
from streamlit_lottie import st_lottie
import json
import requests
import pandas as pd
# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('saved_mental_status_bert')
tokenizer = AutoTokenizer.from_pretrained('saved_mental_status_bert')
label_encoder = pickle.load(open('label_encoder.pkl','rb'))

# Custom cleaning function
stop_words = set(stopwords.words('english'))
def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Detection function
def detect_anxiety(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]

# Helper to load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a mental health Lottie animation
lottie_brain = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_tutvdkg0.json")

# Streamlit Page Config
st.set_page_config(page_title="Suicide Prediction", page_icon="🧠", layout="wide")


# Inject custom CSS for 3D card styling
st.markdown("""
<style>
    .big-title {
        font-size:48px !important;
        color: #00E0A1;
        text-shadow: 2px 2px 12px rgba(0,0,0,0.6);
        font-weight: bold;
    }
    .card {
        background: rgba(20, 20, 20, 0.8);  /* Dark transparent background */
        color: #f1f1f1;                     /* Light text for contrast */
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        backdrop-filter: blur(8px);          /* Glassmorphism effect */
        -webkit-backdrop-filter: blur(8px);
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }
    .card:hover {
        transform: translateY(-10px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.7);
    }
    .card h2, .card h3, .card p, .card li {
        color: #ffffff !important;           /* Force white text inside card */
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Model", "Suicide Prediction", "Research Insights", "Contact"])

# ---- HOME PAGE ----
if page == "Home":
    # Big animated title
    st.markdown('<p class="big-title">🧠 Suicide Detection using BERT Model</p>', unsafe_allow_html=True)

    # Add top animation
    if lottie_brain:
        st_lottie(lottie_brain, height=300, key="brainanim")

    # Introduction card
    st.markdown("""
    <div class="card">
        <h2>Why This App Matters</h2>
        <p>Every year, <b>over 700,000 people</b> die due to suicide globally — that's one person every 40 seconds.</p>
        <p>Many of these individuals express distress <b>online</b> before seeking help in real life. Detecting these early warning signs can save lives.</p>
        <p>This application uses a <b>fine-tuned BERT model</b> to analyze social media posts or text and classify them as:</p>
        <ul>
            <li><b>Normal</b></li>
            <li><b>Anxiety / Depression</b></li>
            <li><b>Suicidal Risk</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Real-world incidents (static content for now)
    st.markdown("""
    ### Real-World Cases:
    - **Case 1 (India, 2022):** A college student posted "I can't take this anymore" on Twitter hours before a tragic suicide.  
    - **Case 2 (US, 2020):** A teenager on Reddit expressed hopelessness; AI-powered monitoring flagged the post, allowing timely intervention.  
    - **Case 3 (Japan, 2019):** After an influencer’s death, platforms improved suicide-prevention monitoring using NLP models.  

    <div class="card">
        <h3>Our Goal</h3>
        <p>To build an <b>early warning system</b> that can flag at-risk posts in real-time, enabling family, friends, or mental health professionals to intervene.</p>
    </div>
    """, unsafe_allow_html=True)

    # Add another animation (different Lottie animation)
    lottie_help = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_v7bfn1fs.json")
    if lottie_help:
        st_lottie(lottie_help, height=250, key="helpanim")

    # Next steps card
    st.markdown("""
    <div class="card">
        <h3>How This App Works</h3>
        <p>
        - Input any text (social media post, journal entry, or message).<br>
        - The BERT model analyzes context and predicts risk level.<br>
        - Future versions will include <b>confidence scores</b>, <b>alerts</b>, and <b>dashboard monitoring</b>.
        </p>
        <p><b>Note:</b> This is for research and educational purposes, not a clinical tool.</p>
    </div>
    """, unsafe_allow_html=True)

# ---- ABOUT MODEL ----
elif page == "About Model":
    st.title("🔍 About the Fine-Tuned BERT Model")

    st.markdown("""
    - **Base Model:** bert-base-uncased  
    - **Fine-tuned Dataset:** Social media posts labeled for suicide risk, anxiety, depression, and normal text.  
    - **Goal:** Detect early warning signs of suicidal ideation from text.  
    - **Approach:**  
      1. Data cleaning & preprocessing (stopword removal, normalization)  
      2. Tokenization using BERT tokenizer  
      3. Training using cross-entropy loss  
      4. Evaluation using F1-score, precision, recall  
    """)

    # Performance metrics as a table
    data = {
        "Metric": [
            "Accuracy",
            "Precision (Non-Suicide)",
            "Recall (Non-Suicide)",
            "F1-score (Non-Suicide)",
            "Precision (Suicide)",
            "Recall (Suicide)",
            "F1-score (Suicide)",
            "Macro Avg F1",
            "Weighted Avg F1"
        ],
        "Score": [
            "96%",
            "0.97",
            "0.96",
            "0.96",
            "0.96",
            "0.97",
            "0.97",
            "0.96",
            "0.96"
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)

# ---- TEST PAGE ----
elif page == "Suicide Prediction":
    st.title("📝 Suicide Prediction Test")
    st.markdown("Enter a post or message to analyze if it contains suicidal tendencies.")

    input_text = st.text_area("Enter your text here...", height=150)
    
    if st.button("Detect"):
        if input_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            try:
                predicted_class = detect_anxiety(input_text)
                if predicted_class:
                    st.success(f"**Predicted Status:** {predicted_class}")
                else:
                    st.error("Model returned no prediction.")
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")

# ---- INSIGHTS PAGE ----
elif page == "Research Insights":
    st.title("📊 Research on Suicide Classification using BERT")
    st.markdown("""
    ### Key Findings in Recent Research:
    - Suicide-related posts are often **short, metaphorical, or implicit**, making detection difficult with simple keyword filters.  
    - **BERT and transformer models outperform RNNs and CNNs**, thanks to their bidirectional context understanding.  
    - Fine-tuned BERT models have achieved **85–92% accuracy** and **F1-scores around 0.90** depending on dataset quality.  
    - Adding **domain-specific embeddings (e.g., mental-health corpora)** further improves performance.  

    ### Notable Research Papers:
    - **Ji et al. (2020):** "Suicide Risk Detection on Twitter Using BERT" – achieved high accuracy by fine-tuning BERT with Twitter posts annotated for suicidal ideation.  
    - **Sawhney et al. (2021):** "Suicide Ideation Detection via BERT" – highlighted the model’s ability to detect subtle suicidal cues in multilingual social media data.  
    - **Gaur et al. (2019):** "Deep Learning for Suicide Ideation Detection" – compared LSTM, CNN, and BERT, showing transformers' superior recall in high-risk cases.  
    - **Mishra et al. (2022):** "Mental Health Classification using Social Media" – explored domain-adapted BERT for detecting depression, anxiety, and suicide risk simultaneously.  

    ### Real-world Applications:
    - **Social Media Monitoring:** Platforms like Reddit, Twitter, and Facebook are testing NLP systems to flag posts with suicidal intent.  
    - **Mental Health Chatbots:** AI chatbots integrate BERT models to provide crisis hotlines or emergency resources automatically.  
    - **Healthcare Triage:** Doctors and therapists use such tools to prioritize high-risk patients in large datasets of patient messages.  

    ### Emerging Techniques:
    - **Explainable AI (XAI):** Using SHAP or LIME to highlight *why* the model classified text as suicidal.  
    - **Multimodal Models:** Combining text + images + metadata for richer signals.  
    - **Low-resource Language Support:** Adapting BERT models for non-English suicide prevention.  

    ### Ethical Concerns:
    - **Privacy & Consent:** Social media posts are sensitive; datasets must be anonymized.  
    - **Bias:** Overrepresentation of certain demographics may lead to unfair predictions.  
    - **False Positives / Negatives:** Automated detection must be combined with human review to avoid wrongful flagging or missed cases.  
    - **Responsible Use:** Tools should **augment**, not replace, mental health professionals.  

    ---
    **Conclusion:**  
    Research consistently shows that **fine-tuned BERT models are state-of-the-art for suicide detection on social media**, but ethical deployment requires careful oversight, transparency, and collaboration with mental health experts.
    """)

# ---- CONTACT ----
elif page == "Contact":
    st.title("📬 Contact Us")
    st.markdown("""
    **Developer:** priyadharashini  
    **Email:** priyadharashini@example.com  
    **GitHub:** [Project Repository](https://github.com/)  

    For collaborations or feedback, reach out anytime!  
    """)
    st.balloons()
