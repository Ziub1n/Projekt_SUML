"""Streamlit app for workplace message appropriateness checker."""

import os
import sys

import streamlit as st

# allow imports from project root when running via `streamlit run app/main.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.predict import load_model, predict_message, MODEL_PATH


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Workplace Message Checker",
    page_icon="💼",
    layout="centered",
)

# ── model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    """Load the pre-trained model from disk."""
    return load_model(MODEL_PATH)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("💼 Workplace Message Checker")
st.write(
    "Paste a message below to check whether it is appropriate "
    "to send in a professional work environment."
)

with st.expander("ℹ️ How it works"):
    st.write(
        "The model was trained on a large dataset of text messages labeled "
        "as toxic or non-toxic. It uses TF-IDF feature extraction combined "
        "with Logistic Regression to classify messages as **Appropriate** or "
        "**Inappropriate** for the workplace."
    )

message = st.text_area(
    label="Your message",
    placeholder="Type or paste your message here...",
    height=150,
)

check_button = st.button("Check message", type="primary", use_container_width=True)

if check_button:
    if not message.strip():
        st.warning("Please enter a message before checking.")
    else:
        with st.spinner("Analysing..."):
            pipeline = get_model()
            result = predict_message(message, pipeline)

        confidence_pct = round(result["confidence"] * 100, 1)

        if result["is_appropriate"]:
            st.success(f"**Appropriate** — confidence {confidence_pct}%")
            st.write("This message appears safe to send in a professional environment.")
        else:
            st.error(f"**Inappropriate** — confidence {confidence_pct}%")
            st.write(
                "This message may contain language that is not suitable "
                "for the workplace. Consider revising it before sending."
            )

        st.progress(result["inappropriate_score"], text="Inappropriateness score")

st.divider()
st.caption("Workplace Message Checker · Built with Streamlit & scikit-learn")
