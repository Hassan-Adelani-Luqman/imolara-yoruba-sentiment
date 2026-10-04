"""Ìmọ̀lára: Yoruba tweet sentiment demo (Gradio, Hugging Face ZeroGPU Space).

Two models trained on AfriSenti Yoruba with diacritic augmentation ("mixed3": every training tweet seen with tone
marks, without tone marks and without any diacritics):
  * AfriBERTa-large fine-tuned (126M parameters), test macro-F1 0.741 ± 0.007
  * word + character TF-IDF + logistic regression, test macro-F1 0.745

Input text goes through exactly the preprocessing used in training (src/data.py: NFC, the AfriSenti test-set
normalisation, optional diacritic removal) before both models score it.

Local run:  IMOLARA_MODEL=models/deploy_afriberta_mixed3 IMOLARA_TFIDF=models/tfidf_mixed3.joblib python app/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE if (HERE / "src").exists() else HERE.parent))   # Space: ./src; repo: ../src
from src.data import LABELS, preprocess   # noqa: E402

MODEL_REPO = os.environ.get("IMOLARA_MODEL", "Hassanadelani1/imolara-afriberta-mixed3")
TFIDF_PATH = os.environ.get("IMOLARA_TFIDF") or hf_hub_download(MODEL_REPO, "tfidf_mixed3.joblib")
GITHUB = "https://github.com/Hassan-Adelani-Luqman/imolara-yoruba-sentiment"

try:                                   # ZeroGPU: a GPU is attached only while a decorated function runs
    import spaces
    gpu = spaces.GPU(duration=15)
    DEVICE = "cuda"
except ImportError:                    # local / CPU
    gpu = lambda fn: fn                # noqa: E731
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).eval().to(DEVICE)
tfidf = joblib.load(TFIDF_PATH)["pipeline"]

FORMS = {"As typed": "original", "Remove tone marks": "no_tones", "Remove all diacritics (tones + under-dots)": "no_diacritics"}


@gpu
def afriberta_probs(text: str) -> list[float]:
    enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return torch.softmax(model(**enc).logits.float(), dim=-1)[0].cpu().tolist()


def classify(text: str, form: str):
    if not text or not text.strip():
        return None, None, "", ""
    seen = preprocess(text, diacritics=FORMS[form])
    if not seen:
        return None, None, "", "_Nothing left after normalisation (links, mentions, emojis and digits are removed)._"
    pa = afriberta_probs(seen)
    pt = tfidf.predict_proba([seen])[0].tolist()
    la, lt = LABELS[max(range(3), key=pa.__getitem__)], LABELS[max(range(3), key=pt.__getitem__)]
    verdict = (f"Both models say **{la}**." if la == lt else
               f"The models **disagree**: AfriBERTa says **{la}**, TF-IDF says **{lt}**. Treat this one with caution.")
    if max(pa) > 0.9 and la != lt:
        verdict += " (AfriBERTa is often over-confident: 94% of its test predictions exceed 0.9 but only ~77% of those are right.)"
    return dict(zip(LABELS, pa)), dict(zip(LABELS, pt)), seen, verdict


EXAMPLES = [
    ["Ẹ kú àbọ̀ o! Inú mi dùn púpọ̀ láti rí yín 😊", "As typed"],
    ["E ku abo o! Inu mi dun pupo lati ri yin", "As typed"],
    ["Ìjọba yìí ti bà wá jẹ́, kò sí iná, kò sí omi.", "As typed"],
    ["Ìpàdé àwọn gómìnà yóò wáyé ní Àbújá lọ́la.", "As typed"],
    ["Mo fẹ́ràn orin yẹn gan an", "Remove all diacritics (tones + under-dots)"],
    ["Omo this traffic no be small o, mi ò ní lè dé ibi iṣẹ́ lásìkò", "As typed"],
    ["Ní àtètèkọ́ṣe Ọlọ́run dá ọ̀run àti ayé", "As typed"],
    ["Àkàrà ti tú sépo", "As typed"],
]

ABOUT = f"""
**What it does.** Classifies the sentiment of a Yoruba tweet as *negative*, *neutral* or *positive* with two models trained on the
AfriSenti-SemEval 2023 Yoruba data (8,522 training tweets). Both were trained on every tweet **with tone marks, without tone marks and
without any diacritics** ("mixed3"), so they keep working when people type Yoruba without diacritics, as most phone keyboards do.

| Model (clean test set, 4,129 tweets) | macro-F1 | weighted-F1 | drop without diacritics |
|---|---|---|---|
| AfriBERTa-large, fine-tuned (126M params) | 0.741 ± 0.007 | 77.5 | −1.2 points |
| TF-IDF word + char n-grams + logistic regression | 0.745 | 77.8 | −0.6 points |
| *same models without diacritic augmentation* | *0.732 / 0.737* | | *−7 to −8 points* |

**How it works.** Your text is normalised exactly like the training data (lower-cased; mentions, links, hashtags, punctuation,
digits and emojis removed; Unicode NFC). The box "What the models see" shows the result. The radio buttons let you remove tone marks
or all diacritics to see the robustness for yourself.

**Known limitations.** Trained on Twitter data only. About 10% of the training labels look unreliable (confident-learning estimate).
Proverbs and idioms, negation (*kò*, *kì í*), and religious or greeting words that appear in non-positive tweets are common error sources.
AfriBERTa's probabilities are over-confident. **Not for decisions about individuals.**

Code, experiments and report: [{GITHUB.removeprefix('https://')}]({GITHUB}) · Model card: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})
"""

with gr.Blocks(title="Ìmọ̀lára: Yoruba sentiment") as demo:
    gr.Markdown("# Ìmọ̀lára · Yoruba tweet sentiment\nType or paste a Yoruba tweet (with or without diacritics) and compare two models.")
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="Yoruba text", lines=3, placeholder="e.g. Ẹ kú àbọ̀ o! Inú mi dùn púpọ̀")
            form = gr.Radio(list(FORMS), value="As typed", label="Simulate typing")
            go = gr.Button("Classify", variant="primary")
        with gr.Column(scale=2):
            seen = gr.Textbox(label="What the models see (after normalisation)", interactive=False)
            verdict = gr.Markdown()
    with gr.Row():
        out_a = gr.Label(label="AfriBERTa-large (fine-tuned, mixed3)", num_top_classes=3)
        out_t = gr.Label(label="TF-IDF + logistic regression (mixed3)", num_top_classes=3)
    gr.Examples(EXAMPLES, inputs=[text, form], label="Examples (the last two are known failure cases: a Bible verse "
                "labelled neutral, and the idiom 'the bean cake has dissolved in the oil', i.e. things fell apart)")
    with gr.Accordion("About the models, results and limitations", open=False):
        gr.Markdown(ABOUT)
    go.click(classify, [text, form], [out_a, out_t, seen, verdict])
    text.submit(classify, [text, form], [out_a, out_t, seen, verdict])

if __name__ == "__main__":
    demo.launch()
