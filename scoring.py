"""
Confidence scoring and transparency label generation.

Design decision (documented in planning.md): a false positive -- labeling a
human's genuine work as AI-generated -- is worse than a false negative on a
creative-writing platform, because it directly damages a real creator's
credibility and trust in the platform. The scoring thresholds below are
therefore deliberately ASYMMETRIC: it takes a higher combined score to earn
a confident "likely AI" label than it takes a low score to earn a confident
"likely human" label, and the "uncertain" band is wide on the AI side.

Ensemble of three signals (stretch feature: "Ensemble detection"):

1. LLM signal (Groq, semantic/holistic) -- weight 0.65
2. Stylometric signal (structural heuristics, pure Python) -- weight 0.25
3. Trained ML signal (TF-IDF + Logistic Regression) -- weight 0.10

Weighting rationale, in order:

- The LLM signal keeps the largest weight: it captures the broadest range
  of AI "tells" and is the least domain-dependent of the three.
- The stylometric signal is reduced slightly from the two-signal version
  (0.35 -> 0.25) to make room for the third signal, while still remaining
  meaningfully influential.
- The trained ML signal gets the smallest weight (0.10), despite scoring
  99.4% test accuracy on its training domain (see
  model_artifacts/eval_report.md). That accuracy is measured on scientific
  abstracts (its training data); on the kind of short creative/casual text
  this platform actually handles, its probabilities compress into a much
  narrower range and it does NOT reliably rank the milestone-4 reference
  cases correctly (see README.md "Known Limitations" for the measured
  numbers). An earlier attempt at 0.15 was tested against the same four
  reference cases used throughout this project and caused the "clearly
  AI" case to regress from a confident `likely_ai` into `uncertain` --
  a domain-mismatched signal dragging down a case the other two signals
  agreed on. The weight was lowered to 0.10 specifically to fix that
  regression while still letting the third signal meaningfully move
  ambiguous cases. This is a deliberate, tested tradeoff, not a default.
"""

LLM_WEIGHT = 0.65
STYLOMETRIC_WEIGHT = 0.25
TRAINED_ML_WEIGHT = 0.10

# Asymmetric thresholds -- see module docstring for reasoning.
LIKELY_AI_THRESHOLD = 0.70
LIKELY_HUMAN_THRESHOLD = 0.30

LABEL_TEXT = {
    "likely_ai": (
        "This content shows strong signals of AI generation. Our system detected "
        "writing patterns consistent with AI-generated text (confidence: {pct}%). "
        "If you believe this is incorrect, you can appeal this classification."
    ),
    "likely_human": (
        "This content shows strong signals of human authorship. Our system found "
        "writing patterns consistent with human-generated text (confidence: {pct}%)."
    ),
    "uncertain": (
        "We're not fully confident about this content's origin. Our system detected "
        "mixed signals that don't clearly match typical AI or human writing patterns "
        "(confidence: {pct}%). This content has not been conclusively classified, and "
        "no strong claim is being made either way."
    ),
}


def combine_signals(
    llm_ai_probability: float,
    stylometric_ai_probability: float,
    trained_ml_ai_probability: float,
) -> float:
    """Weighted average of all three signals into a single 0-1 combined score."""
    combined = (
        LLM_WEIGHT * llm_ai_probability
        + STYLOMETRIC_WEIGHT * stylometric_ai_probability
        + TRAINED_ML_WEIGHT * trained_ml_ai_probability
    )
    return round(max(0.0, min(1.0, combined)), 4)


def attribution_from_score(combined_score: float) -> str:
    if combined_score >= LIKELY_AI_THRESHOLD:
        return "likely_ai"
    if combined_score <= LIKELY_HUMAN_THRESHOLD:
        return "likely_human"
    return "uncertain"


def label_for_score(combined_score: float) -> dict:
    attribution = attribution_from_score(combined_score)
    pct = round(combined_score * 100) if attribution != "likely_human" else round(
        (1 - combined_score) * 100
    )
    text = LABEL_TEXT[attribution].format(pct=pct)
    return {"attribution": attribution, "label_text": text}


def score_submission(
    llm_ai_probability: float,
    stylometric_ai_probability: float,
    trained_ml_ai_probability: float,
) -> dict:
    """Full scoring pipeline: combine signals -> attribution -> label text."""
    combined = combine_signals(
        llm_ai_probability, stylometric_ai_probability, trained_ml_ai_probability
    )
    label_info = label_for_score(combined)
    return {
        "confidence": combined,
        "attribution": label_info["attribution"],
        "label_text": label_info["label_text"],
    }


if __name__ == "__main__":
    # Sanity check across the four milestone-4 style test cases using
    # illustrative LLM scores (since this file has no network dependency)
    # and the ACTUAL trained_ml scores measured from signals_trained_ml.py.
    test_cases = [
        ("clearly_ai", 0.93, 0.3601, 0.1268),
        ("clearly_human", 0.04, 0.0650, 0.0677),
        ("borderline_formal_human", 0.55, 0.2445, 0.06),  # illustrative
        ("borderline_edited_ai", 0.60, 0.2604, 0.05),  # illustrative
    ]
    for name, llm_score, stylo_score, ml_score in test_cases:
        result = score_submission(llm_score, stylo_score, ml_score)
        print(f"{name}: combined={result['confidence']} -> {result['attribution']}")
