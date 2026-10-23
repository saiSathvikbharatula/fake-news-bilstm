"""Streamlit web app for the Bi-LSTM Fake News Detector.

Run:  streamlit run app.py
"""
import json

import streamlit as st

import config

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource(show_spinner="Loading Bi-LSTM model ...")
def get_predictor():
    from src.predict import load_model, predict
    load_model()
    return predict


st.title("📰 Fake News Detector")
st.caption("Bidirectional LSTM + GloVe embeddings, trained on the WELFake dataset")

if not config.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first, then reload this page.")
    st.stop()

predict = get_predictor()

examples = {
    "— choose an example —": "",
    "Likely real": (
        "The Federal Reserve held interest rates steady on Wednesday, saying it would "
        "continue to monitor inflation data before making further changes. Officials "
        "noted that the labor market remained strong while price growth had slowed."
    ),
    "Likely fake": (
        "BREAKING: Scientists CONFIRM that drinking hot water with lemon cures all "
        "diseases and the government is HIDING it from you! Share this before it gets "
        "deleted!!! Doctors are furious about this one simple trick."
    ),
}
choice = st.selectbox("Try an example", list(examples))
text = st.text_area("Paste a news article (headline + body):",
                    value=examples[choice], height=220)

if st.button("Analyze", type="primary", use_container_width=True):
    if len(text.split()) < 5:
        st.warning("Please enter at least a few sentences for a reliable prediction.")
    else:
        try:
            result = predict(text)
        except ValueError as e:
            st.warning(str(e))
        else:
            if result["label"] == "FAKE":
                st.error(f"### 🚩 Likely FAKE  —  {result['confidence']:.1%} confidence")
            else:
                st.success(f"### ✅ Likely REAL  —  {result['confidence']:.1%} confidence")
            st.progress(result["p_fake"], text=f"Probability of fake: {result['p_fake']:.1%}")
            st.caption("This is an ML prediction based on writing patterns, not a fact-check. "
                       "Always verify with trusted sources.")

with st.sidebar:
    st.header("Model performance")
    if config.METRICS_PATH.exists():
        m = json.loads(config.METRICS_PATH.read_text())
        b = m["bilstm"]
        st.metric("Test accuracy", f"{b['accuracy']:.2%}")
        st.metric("F1-score", f"{b['f1']:.4f}")
        st.metric("ROC-AUC", f"{b['roc_auc']:.4f}")
        if m.get("baseline_tfidf_logreg"):
            st.caption(f"Baseline (TF-IDF + LogReg) F1: {m['baseline_tfidf_logreg']['f1']:.4f}")
        st.caption(f"Trained on {m['train_rows']:,} articles")
    for img in ["confusion_matrix.png", "training_curves.png"]:
        p = config.OUTPUT_DIR / img
        if p.exists():
            st.image(str(p))
