"""Ìmọ̀lára: Yoruba tweet sentiment, Streamlit Community Cloud app.

Two models trained on AfriSenti Yoruba with diacritic augmentation ("mixed3"):
  * AfriBERTa-large, fine-tuned (exported to ONNX, int8-quantised; see results/onnx_validation.md)
  * word + character TF-IDF + logistic regression
Both are downloaded from the Hugging Face Hub (Hassanadelani1/imolara-afriberta-mixed3). Input text goes through exactly
the preprocessing used in training (src/data.py). No PyTorch is needed: onnxruntime + tokenizers keep memory under ~1 GB.

Local run:  streamlit run streamlit_app/streamlit_app.py
  (optional overrides: IMOLARA_ONNX, IMOLARA_TOKENIZER, IMOLARA_TFIDF = local file paths)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # repo root, for src/data.py
from src.data import LABELS, preprocess  # noqa: E402

REPO = "Hassanadelani1/imolara-afriberta-mixed3"
GITHUB = "https://github.com/Hassan-Adelani-Luqman/imolara-yoruba-sentiment"
COLORS = {"negative": "#e34948", "neutral": "#8a8984", "positive": "#2a78d6"}
FORMS = {"As typed": "original", "Remove tone marks": "no_tones", "Remove all diacritics": "no_diacritics"}
EXAMPLES = {
    "Positive greeting": "Ẹ kú àbọ̀ o! Inú mi dùn púpọ̀ láti rí yín 😊",
    "Same, typed without diacritics": "E ku abo o! Inu mi dun pupo lati ri yin",
    "Negative complaint": "Ìjọba yìí ti bà wá jẹ́, kò sí iná, kò sí omi.",
    "Neutral news": "Ìpàdé àwọn gómìnà yóò wáyé ní Àbújá lọ́la.",
    "Hard case: code-switched Pidgin complaint": "Omo this traffic no be small o, mi ò ní lè dé ibi iṣẹ́ lásìkò",
    "Hard case: Bible verse (labelled neutral in the data)": "Ní àtètèkọ́ṣe Ọlọ́run dá ọ̀run àti ayé",
    "Hard case: idiom 'the bean cake dissolved in the oil' (things fell apart)": "Àkàrà ti tú sépo",
}


def local_or_hub(env: str, filename: str) -> str:
    return os.environ.get(env) or hf_hub_download(REPO, filename)


@st.cache_resource(show_spinner="Loading the models (first visit only)…")
def load_models():
    tokenizer = Tokenizer.from_file(local_or_hub("IMOLARA_TOKENIZER", "tokenizer.json"))
    tokenizer.enable_truncation(128)                              # as in training
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(local_or_hub("IMOLARA_ONNX", "onnx/model_int8.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    tfidf = joblib.load(local_or_hub("IMOLARA_TFIDF", "tfidf_mixed3.joblib"))["pipeline"]
    return tokenizer, session, tfidf


def afriberta_probs(tokenizer, session, text: str) -> np.ndarray:
    enc = tokenizer.encode(text)
    logits = session.run(None, {"input_ids": np.array([enc.ids], dtype=np.int64),
                                "attention_mask": np.array([enc.attention_mask], dtype=np.int64)})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def prob_chart(probs: np.ndarray) -> alt.Chart:
    df = pd.DataFrame({"label": LABELS, "probability": probs})
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=LABELS, title=None),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None, grid=False)),
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("probability:Q", format=".1%")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("label:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=None))
    text = base.mark_text(align="left", dx=6).encode(text=alt.Text("probability:Q", format=".0%"))
    return (bars + text).properties(height=130)


def show_model(column, name: str, probs: np.ndarray):
    label = LABELS[int(np.argmax(probs))]
    column.markdown(f"**{name}**")
    column.metric("Prediction", label, f"{probs.max():.0%} confidence", delta_color="off")
    column.altair_chart(prob_chart(probs), width="stretch")


st.set_page_config(page_title="Ìmọ̀lára: Yoruba sentiment", page_icon="💬", layout="centered")
st.title("Ìmọ̀lára · Yoruba tweet sentiment")
st.caption("Type or paste a Yoruba tweet, with or without diacritics, and compare a fine-tuned transformer with a "
           "TF-IDF baseline. Both were trained to stay accurate when Yoruba is typed without tone marks.")

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES["Positive greeting"]


def use_example():
    st.session_state.text = EXAMPLES[st.session_state.example]


st.selectbox("Try an example", list(EXAMPLES), key="example", on_change=use_example)
text = st.text_area("Yoruba text", key="text", height=100)
form = st.radio("Simulate typing", list(FORMS), horizontal=True)

tokenizer, session, tfidf = load_models()
seen = preprocess(text or "", diacritics=FORMS[form])
if not seen:
    st.info("Nothing left after normalisation: links, mentions, hashtags, emojis and digits are removed. Type some Yoruba words.")
else:
    pa = afriberta_probs(tokenizer, session, seen)
    pt = tfidf.predict_proba([seen])[0]
    left, right = st.columns(2)
    show_model(left, "AfriBERTa-large (fine-tuned, mixed3)", pa)
    show_model(right, "TF-IDF + logistic regression (mixed3)", pt)
    la, lt = LABELS[int(pa.argmax())], LABELS[int(pt.argmax())]
    if la == lt:
        st.success(f"Both models say **{la}**.")
    else:
        st.warning(f"The models disagree: AfriBERTa says **{la}**, TF-IDF says **{lt}**. Treat this prediction with caution.")
    st.markdown("**What the models see** (after the same normalisation as the training data):")
    st.code(seen, language=None)

with st.expander("About the models, results and limitations"):
    st.markdown(f"""
Both models were trained on the **AfriSenti-SemEval 2023** Yoruba tweets (8,522 training tweets), with every tweet included
**with tone marks, without tone marks and without any diacritics** ("mixed3").

| Model (clean test set, 4,129 tweets) | macro-F1 | weighted-F1 | drop without diacritics |
|---|---|---|---|
| AfriBERTa-large, fine-tuned (126M parameters) | 0.741 ± 0.007 | 77.5 | −1.2 points |
| TF-IDF word + char n-grams + logistic regression | 0.745 | 77.8 | −0.6 points |
| *same models without diacritic augmentation* | *0.732 / 0.737* | | *−7 to −8 points* |

The web app runs AfriBERTa as an **int8-quantised ONNX** model (4× smaller, no PyTorch needed); its agreement with the original model is
reported in the repository (`results/onnx_validation.md`).

**Known limitations.** Trained on Twitter data only. About 10% of the training labels look unreliable (confident-learning estimate).
Proverbs and idioms, negation (*kò*, *kì í*), and religious or greeting words in non-positive tweets are common error sources.
AfriBERTa's probabilities are over-confident. **Not for decisions about individuals.**

Code, experiments and report: [{GITHUB.removeprefix("https://")}]({GITHUB}) · Model card: [{REPO}](https://huggingface.co/{REPO})
""")
