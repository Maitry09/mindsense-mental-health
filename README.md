# 🧠 MindSense — AI-Powered Mental Health Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![BERT](https://img.shields.io/badge/Model-BERT-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-89%25-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> An NLP-based mental health detection system that classifies text into 7 mental health categories using fine-tuned BERT, with explainable AI (SHAP) and a live web interface.

---

## 🌐 Live Demo

👉 **[Try MindSense Live](https://mindsense.streamlit.app)**

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [What Makes This Different](#-what-makes-this-different)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Explainability — SHAP](#-explainability--shap)
- [Tech Stack](#-tech-stack)
- [How to Run Locally](#-how-to-run-locally)
- [Screenshots](#-screenshots)
- [Ethical Considerations](#-ethical-considerations)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🎯 Problem Statement

Mental health conditions affect **1 in 4 people globally**, yet early detection remains a major challenge. Most people do not seek help due to stigma or lack of awareness. This project builds an AI system that can analyze text written by a person and detect potential mental health conditions — enabling early intervention and awareness.

---

## ✨ What Makes This Different

| Feature | This Project | Typical Student Project |
|---|---|---|
| Model | Fine-tuned BERT | Basic Logistic Regression |
| Classes | 7 mental health categories | Binary (depressed / not) |
| Explainability | SHAP word importance | None |
| Class imbalance fix | Augmentation + class weights | None |
| Deployment | Live web app | Jupyter notebook only |
| Dataset size | 53,000+ real Reddit posts | Small CSV |

---

## 📊 Dataset

- **Source:** [Kaggle — Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- **Size:** 53,000+ Reddit posts
- **Format:** CSV with `statement` (text) and `status` (label) columns
- **Categories:** Normal, Depression, Anxiety, Bipolar, PTSD, Stress, Personality Disorder

### Class Distribution

| Category | Description |
|---|---|
| Normal | No significant mental health concern |
| Depression | Persistent sadness, hopelessness |
| Anxiety | Excessive worry, nervousness |
| Bipolar | Extreme mood swings |
| PTSD | Trauma-related flashbacks |
| Stress | Overwhelm from external pressures |
| Personality Disorder | Enduring patterns affecting relationships |

---

## 📁 Project Structure

```
mindsense-mental-health/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── notebooks/
│   ├── data_preprocessing_and_basic-ml.ipynb   # Data cleaning and EDA and  LR, RF, SVM baseline
│   ├── mental_health_complete_pipeline  # BERT fine-tuning (Google Colab)
│
├── outputs/
│   ├── class_distribution.png      # Class balance chart
|   ├── model_comparision.png       # Comparisions
│   ├── wordclouds.png              # Word clouds per category
│   ├── confusion_matrix.png        # Model confusion matrix
│   ├── before_after_f1.png         # Class improvement chart
│   └── shap_explanation.png        # SHAP word importance
│
└── mentalbert_model/                  # Saved model (on HuggingFace)
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── model.safetensors
    └── label_encoder.pkl
```

---

## 🔬 Methodology

### Phase 1 — Data Cleaning & EDA
- Removed URLs, special characters, duplicates
- Handled missing values
- Generated word clouds per category
- Analyzed text length distribution

### Phase 2 — Baseline ML Models
- Converted text to numbers using **TF-IDF** (10,000 features, unigrams + bigrams)
- Trained Logistic Regression, Random Forest, SVM
- Evaluated using accuracy, F1-score, confusion matrix

### Phase 3 — BERT Fine-tuning
- Used `bert-base-uncased` from HuggingFace Transformers
- Fine-tuned on the full 53,000+ post dataset
- Trained on Google Colab T4 GPU (free)
- 3 epochs, learning rate 1e-5, batch size 16

### Phase 4 — Class Imbalance Fix
- Identified weak classes: Suicidal, Stress, Personality Disorder
- Applied **3 augmentation techniques**: word swap, random deletion, key phrase duplication
- Applied **custom class weights** with manual boost for weak classes
- Retrained model — significant F1 improvement on weak classes

### Phase 5 — Explainability (SHAP)
- Used `shap.Explainer` with Text masker
- Generated word-level importance scores per prediction
- Visualized which words drove each mental health prediction

---

## 📈 Model Performance

### Accuracy Comparison

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~78% | TF-IDF features |
| Random Forest | ~74% | TF-IDF features |
| SVM (LinearSVC) | ~82% | Best baseline |
| BERT v1 | ~83% | Fine-tuned, 3 epochs |
| **BERT v2 (final)** | **~87%** | Augmentation + class weights |

### Per-Class F1 Score (BERT v2)

| Category | F1 Score |
|---|---|
| Normal | 0.91 |
| Depression | 0.89 |
| Anxiety | 0.86 |
| Bipolar | 0.84 |
| PTSD | 0.87 |
| Stress | 0.82 |
| Personality Disorder | 0.80 |

---

## 🔍 Explainability — SHAP

This project uses **SHAP (SHapley Additive exPlanations)** to explain every prediction:

- 🔴 **Red words** = pushed the model toward the predicted category
- 🔵 **Blue words** = pushed the model away from the prediction

This makes the model transparent and trustworthy — a key requirement for any real-world healthcare AI application.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | PyTorch, HuggingFace Transformers |
| ML | Scikit-learn |
| Explainability | SHAP |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Web App | Streamlit |
| Model Hosting | HuggingFace Hub |
| Deployment | Streamlit Cloud |
| Training | Google Colab (free T4 GPU) |

---

## 💻 How to Run Locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/Maitry09/mindsense-mental-health.git
cd mindsense-mental-health
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set up HuggingFace token
Create a file `.streamlit/secrets.toml` and add:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Test with these sample sentences
```
I feel completely hopeless and empty, nothing makes me happy anymore
I keep having flashbacks and nightmares about what happened
My mood changes drastically — sometimes I feel invincible, then worthless
I feel fine today, had a great day with my family
I want to end everything, I see no point in living anymore
```

---

## ⚖️ Ethical Considerations

- This tool is for **educational purposes only**
- It is **not a substitute** for professional mental health diagnosis
- For **Suicidal** predictions, the app displays India crisis helplines:
  - **iCall:** 9152987821
  - **Vandrevala Foundation:** 1860-2662-345 (24x7, free)
- Training data is from public Reddit posts — no personal data collected
- Model predictions should never be used to make clinical decisions

---

## 🚀 Future Work

- Add voice input — detect mental health from speech patterns
- Integrate physiological signals (HRV, EEG) for multi-modal detection
- Build personalized baseline using continual learning
- Add multilingual support for Hindi and regional languages
- Implement federated learning for on-device privacy-preserving inference
- Deploy on AWS/Azure for production-grade reliability

---

## 👤 Author

**Maitry**
- GitHub: [@Maitry09](https://github.com/Maitry09)
- HuggingFace: [@maitry30](https://huggingface.co/maitry30)
- Live App: [mindsense.streamlit.app](https://mindsense.streamlit.app)

---

## 📄 License

This project is licensed under the MIT License.

---

> ⭐ If you found this project useful, please give it a star on GitHub!
