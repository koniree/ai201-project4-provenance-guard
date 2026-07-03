"""
Signal 3: trained ML classifier (TF-IDF + Logistic Regression).

Trained on the AI-GA dataset (28,662 balanced samples of human vs.
GPT-3-generated scientific abstracts; see training/download_data.py and
training/train.py). Captures learned lexical patterns -- both genuine
AI-framing tells ("this paper examines/presents/investigates...") and,
honestly, some scientific-abstract formatting artifacts specific to the
training domain (see model_artifacts/eval_report.md and README.md "Known
Limitations" for the full breakdown of what the model actually learned).

In-domain performance is very strong (99.4% test accuracy). Out-of-domain
performance -- on the kind of short creative/casual text this platform
actually handles -- is measurably weaker and was NOT papered over with an
ad-hoc calibration fit to a handful of samples; see README for the actual
numbers. This signal is included in the ensemble at a deliberately modest
weight for that reason (see scoring.py).

If the artifacts are missing (model not yet trained), this signal
degrades gracefully to a neutral 0.5, the same pattern used by the LLM
signal for its own failure mode.
"""

import os

import joblib

HERE = os.path.dirname(__file__)
VECTORIZER_PATH = os.path.join(HERE, "model_artifacts", "tfidf_vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(HERE, "model_artifacts", "logreg_classifier.joblib")

_vectorizer = None
_classifier = None
_load_error = None


def _load_artifacts():
    global _vectorizer, _classifier, _load_error
    if _vectorizer is not None and _classifier is not None:
        return
    try:
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)


def trained_ml_signal(text: str) -> dict:
    _load_artifacts()

    if _vectorizer is None or _classifier is None:
        return {
            "trained_ml_ai_probability": 0.5,
            "error": _load_error or "Model artifacts not found. Run training/train.py first.",
        }

    x = _vectorizer.transform([text])
    prob_ai = float(_classifier.predict_proba(x)[0][1])

    return {
        "trained_ml_ai_probability": round(prob_ai, 4),
        "error": None,
    }


if __name__ == "__main__":
    samples = {
        "clearly_ai": (
            "Artificial intelligence represents a transformative paradigm shift in modern society. "
            "It is important to note that while the benefits of AI are numerous, it is equally "
            "essential to consider the ethical implications."
        ),
        "clearly_human": (
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium in it."
        ),
    }
    for name, txt in samples.items():
        print(name, "->", trained_ml_signal(txt))
