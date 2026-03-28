# app.py — MindSense Mental Health Detection (Final Version)

import streamlit as st
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import matplotlib.pyplot as plt
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MindSense — Mental Health Detection",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-size: 16px;
        padding: 10px;
    }
    .stTextArea textarea { border-radius: 10px; }

    /* Hide everything Streamlit branded */
    footer { visibility: hidden !important; }
    footer * { visibility: hidden !important; }
    header { visibility: hidden !important; }
    header * { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    #MainMenu * { visibility: hidden !important; }
    .stDeployButton { display: none !important; }

    /* Hide GitHub username and Streamlit badge bottom right */
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    ._profileContainer_gzau3_53 { display: none !important; }
    ._link_gzau3_10 { display: none !important; }
    ._profilePreview_gzau3_63 { display: none !important; }

    /* Hide bottom bar completely */
    .reportview-container .main footer { display: none !important; }
    section[data-testid="stSidebar"] div[class*="profileContainer"] {
        display: none !important;
    }
    div[class*="ProfileContainer"] { display: none !important; }
    div[class*="viewerBadge"] { display: none !important; }
    div[class*="streamlitBadge"] { display: none !important; }
    a[href*="streamlit.io"] { display: none !important; }
    a[href*="github.com"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model once ───────────────────────────────────────────
@st.cache_resource
def load_model():
    HF_REPO  = "maitry30/mindsense-bert"
    hf_token = st.secrets.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO,
        token=hf_token
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO,
        token=hf_token,
        ignore_mismatched_sizes=True
    )
    le_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="label_encoder.pkl",
        token=hf_token
    )
    le = joblib.load(le_path)
    model.eval()
    return tokenizer, model, le

tokenizer, model, le = load_model()

device = torch.device('cpu')
model  = model.to(device)

# ── Prediction function ───────────────────────────────────────
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
    pred_idx   = torch.argmax(probs).item()
    pred_label = le.classes_[pred_idx]
    confidence = probs[0][pred_idx].item()
    all_probs  = probs[0].numpy()
    return pred_label, confidence, all_probs

# ── SHAP explainer ────────────────────────────────────────────
@st.cache_resource
def load_explainer(_tokenizer, _model):
    def predict_proba(texts):
        inputs = _tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=128, return_tensors='pt'
        )
        with torch.no_grad():
            out   = _model(**inputs)
            probs = torch.softmax(out.logits, dim=1)
        return probs.numpy()

    explainer = shap.Explainer(
        predict_proba,
        masker=shap.maskers.Text(_tokenizer)
    )
    return explainer

# ── Color map ─────────────────────────────────────────────────
COLORS = {
    "Normal":               "#4CAF50",
    "Depression":           "#2196F3",
    "Anxiety":              "#FF9800",
    "Bipolar":              "#9C27B0",
    "Stress":               "#FF5722",
    "Suicidal":             "#B71C1C",
    "Personality disorder": "#00BCD4"
}

# ── Condition info shown after prediction ─────────────────────
INFO = {
    "Normal":               "No significant mental health concerns detected.",
    "Depression":           "Feelings of persistent sadness and loss of interest.",
    "Anxiety":              "Excessive worry, nervousness or unease.",
    "Bipolar":              "Extreme mood swings between highs and lows.",
    "Stress":               "Overwhelm from external pressures or demands.",
    "Suicidal":             "Please seek immediate help. You are not alone.",
    "Personality disorder": "Enduring patterns affecting thinking and relationships."
}

# ── Header ────────────────────────────────────────────────────
st.title("🧠 MindSense")
st.subheader("AI-Powered Mental Health Text Analysis")
st.markdown(
    "Type how you are feeling below. "
    "The model will analyze your text and explain its reasoning."
)
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────
user_input = st.text_area(
    "Enter your text here:",
    placeholder="e.g. I have been feeling very low and hopeless lately...",
    height=160
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("Analyze Text", type="primary")

show_shap = st.checkbox("Show word-level explanation (SHAP)", value=True)

# ── Output ────────────────────────────────────────────────────
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter some text first.")

    elif len(user_input.split()) < 5:
        st.warning("Please enter at least 5 words for accurate prediction.")

    else:
        with st.spinner("Analyzing..."):
            pred_label, confidence, all_probs = predict(user_input)

        # ── Result card ───────────────────────────────────────
        color = COLORS.get(pred_label, "#607D8B")
        info  = INFO.get(pred_label, "")

        st.markdown(f"""
        <div style='background:{color}18; border-left:6px solid {color};
                    padding:20px; border-radius:10px; margin:20px 0'>
            <h2 style='color:{color}; margin:0'>Detected: {pred_label}</h2>
            <p style='margin:6px 0 4px; font-size:16px; color:#444'>
                Confidence: <b>{confidence*100:.1f}%</b>
            </p>
            <p style='margin:0; font-size:14px; color:#666'>{info}</p>
        </div>
        """, unsafe_allow_html=True)

        # Crisis message for suicidal
        if pred_label == "Suicidal":
            st.error(
                "If you or someone you know is in crisis, "
                "please call iCall: 9152987821 or Vandrevala Foundation: 1860-2662-345 "
                "(India, 24x7 free helpline)"
            )

        # ── Probability chart ──────────────────────────────────
        st.markdown("#### Confidence across all categories")
        fig, ax = plt.subplots(figsize=(8, 4))
        bar_colors = [COLORS.get(c, '#607D8B') for c in le.classes_]
        bars = ax.barh(
            le.classes_, all_probs * 100,
            color=bar_colors, edgecolor='white', height=0.6
        )
        for bar, prob in zip(bars, all_probs):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{prob*100:.1f}%', va='center', fontsize=10
            )
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 115)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)
        plt.tight_layout()
        st.pyplot(fig)

        # ── SHAP explanation ───────────────────────────────────
        if show_shap:
            st.markdown("#### Why did the model predict this?")
            with st.spinner("Generating word explanation (takes ~30 seconds)..."):
                try:
                    explainer  = load_explainer(tokenizer, model)
                    shap_vals  = explainer([user_input])
                    pred_idx   = np.argmax(all_probs)
                    words      = shap_vals[0].data
                    values     = shap_vals[0].values[:, pred_idx]

                    word_shap  = sorted(
                        zip(words, values),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:12]

                    words_plot  = [w[0] for w in word_shap]
                    values_plot = [w[1] for w in word_shap]
                    shap_colors = ['#E53935' if v > 0 else '#1E88E5'
                                   for v in values_plot]

                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    ax2.barh(
                        words_plot[::-1], values_plot[::-1],
                        color=shap_colors[::-1]
                    )
                    ax2.axvline(x=0, color='black', linewidth=0.8)
                    ax2.set_title(
                        f'Words driving "{pred_label}" prediction\n'
                        'Red = strong signal  |  Blue = against prediction',
                        fontsize=11, fontweight='bold'
                    )
                    ax2.spines[['top', 'right']].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig2)

                except Exception as e:
                    st.info("SHAP explanation unavailable for this input.")

        # ── Disclaimer ─────────────────────────────────────────
        st.info(
            "Disclaimer: This tool is for educational purposes only "
            "and is not a substitute for professional mental health advice. "
            "If you are struggling please reach out to a mental health professional."
        )

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About MindSense")
    st.markdown(
        "MindSense uses a fine-tuned BERT model trained "
        "on 53,000+ real mental health posts from Reddit."
    )
    st.markdown("---")
    st.markdown("**Model Details**")
    st.markdown("- Model: BERT base uncased")
    st.markdown("- Accuracy: ~89%")
    st.markdown("- Classes: 7 categories")
    st.markdown("- Dataset: Kaggle (53k+ posts)")
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("Python · PyTorch · HuggingFace")
    st.markdown("Scikit-learn · SHAP · Streamlit")
    st.markdown("---")
    st.markdown("**Categories**")
    for cat, col in COLORS.items():
        st.markdown(
            f"<span style='color:{col}'>■</span> {cat}",
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.markdown("**Crisis Helplines (India)**")
    st.markdown("iCall: 9152987821")
    st.markdown("Vandrevala: 1860-2662-345")