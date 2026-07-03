"""
Reproduces two analyses referenced in README.md "Known Limitations":

1. Top TF-IDF features driving the trained classifier's predictions --
   checks whether it learned genuine AI-writing tells vs. dataset-specific
   formatting artifacts.
2. Out-of-domain generalization check -- runs the trained classifier on
   the project's actual target domain (short creative/casual text) using
   the same four milestone-4 reference inputs used throughout this
   project, instead of its scientific-abstract training domain.

Run after training/train.py has produced model_artifacts/*.joblib.
"""

import os

import joblib
import numpy as np

HERE = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "model_artifacts")

REFERENCE_CASES = {
    "clearly_ai": (
        "Artificial intelligence represents a transformative paradigm shift in modern society. "
        "It is important to note that while the benefits of AI are numerous, it is equally "
        "essential to consider the ethical implications. Furthermore, stakeholders across "
        "various sectors must collaborate to ensure responsible deployment."
    ),
    "clearly_human": (
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in it and "
        "i was thirsty for like three hours after. my friend got the spicy version and "
        "said it was better. probably won't go back unless someone drags me there"
    ),
    "borderline_formal_human": (
        "The relationship between monetary policy and asset price inflation has been "
        "extensively studied in the literature. Central banks face a fundamental tension "
        "between their mandate for price stability and the unintended consequences of "
        "prolonged low interest rates on equity and real estate valuations."
    ),
    "borderline_edited_ai": (
        "I've been thinking a lot about remote work lately. There are genuine tradeoffs -- "
        "flexibility and no commute on one side, isolation and blurred work-life boundaries "
        "on the other. Studies show productivity varies widely by individual and role type."
    ),
}


def main():
    vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib"))
    classifier = joblib.load(os.path.join(ARTIFACTS_DIR, "logreg_classifier.joblib"))

    print("=" * 60)
    print("TOP TF-IDF FEATURES BY PREDICTION DIRECTION")
    print("=" * 60)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]

    top_ai_idx = np.argsort(coefs)[-15:][::-1]
    top_human_idx = np.argsort(coefs)[:15]

    print("\n-> AI-generated (label=1):")
    for i in top_ai_idx:
        print(f"   {feature_names[i]:30s} {coefs[i]:.3f}")

    print("\n-> Human-written (label=0):")
    for i in top_human_idx:
        print(f"   {feature_names[i]:30s} {coefs[i]:.3f}")

    print()
    print("=" * 60)
    print("OUT-OF-DOMAIN CHECK (creative/casual text, not scientific abstracts)")
    print("=" * 60)
    for name, text in REFERENCE_CASES.items():
        x = vectorizer.transform([text])
        prob_ai = classifier.predict_proba(x)[0][1]
        print(f"   {name:28s} -> P(AI) = {prob_ai:.4f}")


if __name__ == "__main__":
    main()
