# Suicide Prediction Using BERT

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)  
[![Transformers](https://img.shields.io/badge/Transformers-4.56+-yellow.svg)](https://huggingface.co/transformers/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

A deep learning project that leverages **BERT (Bidirectional Encoder Representations from Transformers)** to classify and predict suicide-related content from text data. This project is aimed at supporting mental health awareness by enabling early detection of potentially harmful or self-harming statements.

---

## 🚀 Features
- **Transformer-based model:** Fine-tuned `bert-base-uncased` for suicide detection.  
- **High performance:** Achieved **96% accuracy** with strong precision/recall balance.  
- **Robust preprocessing:** Handles noise, punctuation, and stopwords.  
- **Easy deployment:** Built to integrate with APIs or web apps (Streamlit-ready).  

---

## 📊 Model Performance
- **Accuracy:** 96%  
- **Precision (Non-Suicide):** 0.97  
- **Recall (Non-Suicide):** 0.96  
- **F1-score (Non-Suicide):** 0.96  
- **Precision (Suicide):** 0.96  
- **Recall (Suicide):** 0.97  
- **F1-score (Suicide):** 0.96  

These metrics demonstrate a balanced trade-off between **false positives** and **false negatives**, making the model reliable for real-world applications.  

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+  
- **Libraries:**  
  - [Transformers](https://huggingface.co/transformers/) (Hugging Face)  
  - [PyTorch](https://pytorch.org/)  
  - pandas, numpy  
  - scikit-learn (metrics, preprocessing)  
  - streamlit (for deployment)  

---

## 📂 Project Structure
suicide-prediction-bert/
│
├── data/ # Dataset (not included in repo for privacy)
├── models/ # Saved model weights
├── notebooks/ # Jupyter notebooks for EDA and training
├── src/
│ ├── preprocessing.py # Data cleaning utilities
│ ├── dataset.py # PyTorch dataset loader
│ ├── train.py # Model training script
│ ├── evaluate.py # Evaluation and metrics
│ └── predict.py # Prediction interface
├── app.py # Streamlit demo app
└── requirements.txt # Dependencies

---
## ⚙️ Dataset link and full project like 
-G-Drive link: https://drive.google.com/file/d/1Wh1UTKWXtcQr-ad3CG-nmEipYfRAvYCE/view?usp=sharing  

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/suicide-prediction-bert.git
cd suicide-prediction-bert
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
run this code using this command like
streamlit run main.py


---

Do you also want me to **generate a sample `requirements.txt` file** with exact library versions to match BERT, PyTorch, and Streamlit? Or keep it minimal?


