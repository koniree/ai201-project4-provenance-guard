"""
Signal 2: Stylometric heuristics.

Captures measurable statistical properties of writing that tend to differ
between human and AI-generated text: sentence length variance, vocabulary
diversity (type-token ratio), and punctuation density.

What this signal captures: surface-level structural uniformity vs.
irregularity. AI-generated text (especially from base instruction-tuned
models with default sampling) tends toward more uniform sentence lengths,
more "balanced" punctuation, and less lexical repetition/idiosyncrasy than
typical human writing.

What it can't capture: meaning, coherence, factual grounding, or intent.
A human writer with a very disciplined, formal style (e.g. academic or
legal writing) can score similarly to AI text on these metrics -- this is
a known blind spot documented in planning.md. It also can't detect
AI-generated text that has been lightly edited by a human to break up
uniform sentence structure.
"""

import re
import statistics


def _split_sentences(text: str):
    # Simple sentence splitter -- good enough for heuristic purposes.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _tokenize_words(text: str):
    return re.findall(r"[A-Za-z']+", text.lower())


def sentence_length_variance_score(text: str) -> float:
    """
    Returns a 0-1 score where HIGHER = more uniform sentence lengths
    (more AI-like). Human writing tends to have higher variance.
    """
    sentences = _split_sentences(text)
    lengths = [len(_tokenize_words(s)) for s in sentences if _tokenize_words(s)]

    if len(lengths) < 2:
        return 0.5  # not enough signal, stay neutral

    mean_len = statistics.mean(lengths)
    stdev_len = statistics.pstdev(lengths)

    if mean_len == 0:
        return 0.5

    coeff_of_variation = stdev_len / mean_len  # normalized spread

    # Empirically, human writing often has CoV > 0.5; very uniform
    # (AI-like) text often has CoV < 0.3. Map CoV inversely onto a
    # 0-1 "AI-likeness" score, clamped.
    score = 1.0 - min(coeff_of_variation / 0.7, 1.0)
    return max(0.0, min(1.0, score))


def type_token_ratio_score(text: str) -> float:
    """
    Returns a 0-1 score where HIGHER = lower vocabulary diversity
    (more AI-like). AI text often reuses safe, common vocabulary;
    human text (especially casual writing) tends to be lexically
    "messier" -- slang, repetition for emphasis, tangents.
    """
    words = _tokenize_words(text)
    if len(words) < 5:
        return 0.5

    ttr = len(set(words)) / len(words)

    # Longer texts naturally have lower raw TTR, so this is a coarse
    # heuristic best applied to short-to-medium excerpts (which is the
    # expected input for this project: poems, blog posts, short stories).
    # Typical human casual writing: TTR ~0.65-0.85 for short passages.
    # Typical AI writing: TTR ~0.55-0.70 for short passages (more repetition
    # of transitional/hedging phrases like "it is important to note").
    score = 1.0 - min(max(ttr - 0.5, 0.0) / 0.4, 1.0)
    return max(0.0, min(1.0, score))


def punctuation_density_score(text: str) -> float:
    """
    Returns a 0-1 score where HIGHER = more "balanced"/formal punctuation
    density (more AI-like). AI text tends to use commas, semicolons, and
    transitional punctuation more consistently; human casual text tends
    to under- or over-punctuate irregularly (run-ons, missing commas,
    excess exclamation points, ellipses).
    """
    words = _tokenize_words(text)
    if not words:
        return 0.5

    commas = text.count(",")
    semicolons = text.count(";")
    exclamations = text.count("!")
    ellipses = text.count("...")

    formal_density = (commas + semicolons) / len(words)
    informal_markers = (exclamations + ellipses) / max(len(words), 1)

    # High formal density + low informal markers => more AI-like.
    score = min(formal_density / 0.08, 1.0) * (1.0 - min(informal_markers / 0.03, 1.0))
    return max(0.0, min(1.0, score))


def stylometric_signal(text: str) -> dict:
    """
    Combines three stylometric metrics into a single signal score in [0, 1],
    where higher = more consistent with AI-generated text.

    Equal-weighted average of the three sub-metrics. Each sub-metric is
    independently interpretable, which is useful for debugging disagreement
    between them (see README known limitations).
    """
    sl_score = sentence_length_variance_score(text)
    ttr_score = type_token_ratio_score(text)
    punct_score = punctuation_density_score(text)

    combined = (sl_score + ttr_score + punct_score) / 3.0

    return {
        "stylometric_ai_probability": round(combined, 4),
        "components": {
            "sentence_length_uniformity": round(sl_score, 4),
            "low_vocabulary_diversity": round(ttr_score, 4),
            "formal_punctuation_density": round(punct_score, 4),
        },
    }


if __name__ == "__main__":
    samples = {
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

    for name, txt in samples.items():
        print(name, "->", stylometric_signal(txt))
