"""
Trains the learned ML signal: TF-IDF features + Logistic Regression,
classifying text as AI-generated (1) vs human-written (0).

Run:
    python training/download_data.py   # fetches ai-ga-dataset.csv if not present
    python training/train.py

Outputs (written to model_artifacts/):
    - tfidf_vectorizer.joblib
    - logreg_classifier.joblib
    - eval_report.md   (metrics + confusion matrix, for documentation)

Also trains a LinearSVC as a comparison point (see eval_report.md) --
Logistic Regression is used in the shipped signal because it outputs
calibrated-ish probabilities via predict_proba, which the rest of the
pipeline needs; LinearSVC would need extra calibration (CalibratedClassifierCV)
to produce a usable probability.
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "ai-ga-dataset.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(HERE), "model_artifacts")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "eval_report.md")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["abstract", "label"])
    df["text"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).str.strip()
    return df[["text", "label"]]


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Total samples: {len(df)}  |  label balance:\n{df['label'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing (TF-IDF, unigrams + bigrams, max 20k features)...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    logreg.fit(X_train_vec, y_train)

    print("Training LinearSVC (comparison only, not shipped)...")
    svc = LinearSVC(class_weight="balanced")
    svc.fit(X_train_vec, y_train)

    report_lines = ["# Trained ML Signal -- Evaluation Report\n"]
    report_lines.append(f"Dataset: `ai-ga-dataset.csv` ({len(df)} samples, "
                         f"{y_train.shape[0]} train / {y_test.shape[0]} test, stratified split)\n")

    for name, model, needs_decision in [
        ("Logistic Regression (shipped)", logreg, False),
        ("LinearSVC (comparison)", svc, True),
    ]:
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        auc = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_vec)[:, 1]
            auc = roc_auc_score(y_test, probs)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_vec)
            auc = roc_auc_score(y_test, scores)

        cm = confusion_matrix(y_test, preds)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}"
              + (f"  ROC-AUC: {auc:.4f}" if auc is not None else ""))
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, target_names=["human", "ai"]))

        report_lines.append(f"\n## {name}\n")
        report_lines.append(f"- Accuracy: **{acc:.4f}**")
        report_lines.append(f"- Precision: **{prec:.4f}**")
        report_lines.append(f"- Recall: **{rec:.4f}**")
        report_lines.append(f"- F1: **{f1:.4f}**")
        if auc is not None:
            report_lines.append(f"- ROC-AUC: **{auc:.4f}**")
        report_lines.append("\nConfusion matrix (rows=true, cols=predicted, order=[human, ai]):\n")
        report_lines.append("```")
        report_lines.append(str(cm))
        report_lines.append("```\n")
        report_lines.append("```")
        report_lines.append(classification_report(y_test, preds, target_names=["human", "ai"]))
        report_lines.append("```\n")

    joblib.dump(vectorizer, os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(logreg, os.path.join(ARTIFACTS_DIR, "logreg_classifier.joblib"))

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved model artifacts to {ARTIFACTS_DIR}/")
    print(f"Saved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
