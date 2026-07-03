# Provenance Guard — Planning

## Architecture Narrative

A piece of text enters the system through `POST /submit` along with a
`creator_id`. The Flask app hands the raw text to two independent detection
signals: the LLM signal (Groq, `llama-3.3-70b-versatile`) makes a holistic
semantic judgment about whether the writing "reads" as human or AI, while
the stylometric signal computes three structural metrics in pure Python
(sentence length uniformity, vocabulary diversity, punctuation density) and
combines them into a second score. Both signals return a 0–1 "AI
probability." The scoring module combines them with a weighted average
(LLM weighted higher, since it's the more holistic signal) into a single
confidence score, which is mapped through asymmetric thresholds into one of
three attributions: `likely_ai`, `likely_human`, or `uncertain`. That
attribution selects one of three fixed transparency label templates, which
is filled in with the actual confidence percentage. The full result —
both raw signal scores, the combined score, the attribution, and the label
— is written to a structured SQLite audit log row keyed by a generated
`content_id`, and returned to the caller.

If a creator disputes a classification, they call `POST /appeal` with the
`content_id` and their reasoning. The system looks up the existing
submission row, updates its `status` to `under_review`, and logs the
appeal reasoning and timestamp alongside the original decision in the same
row — no new classification is triggered. `GET /log` exposes recent rows
(including appealed ones) as structured JSON for review and grading.

## Architecture Diagram

```
                         SUBMISSION FLOW
                         ================

   creator/platform
         |
         | POST /submit { text, creator_id }
         v
   +----------------+
   |  Flask route   |
   |   /submit      |
   +----------------+
         |
         | raw text
         |------------------+------------------+
         v                  v                  v
 +----------------+ +------------------+ +----------------------+
 | Signal 1: LLM  | | Signal 2: Stylo  | | Signal 3: Trained ML  |
 | (Groq API)     | | heuristics       | | (TF-IDF + LogReg,     |
 | -> ai_prob 0-1 | | (pure Python)    | |  trained offline on   |
 |                | | -> ai_prob 0-1   | |  AI-GA dataset)        |
 |                | |                  | | -> ai_prob 0-1        |
 +----------------+ +------------------+ +----------------------+
         |                  |                     |
         | llm_score        | stylo_score         | trained_ml_score
         +------------------+----------+----------+
                                        v
                          +------------------------+
                          |  scoring.py            |
                          |  weighted combine       |
                          |  (0.65 / 0.25 / 0.10)  |
                          |  -> confidence 0-1      |
                          +------------------------+
                         |
                         | confidence
                         v
              +-----------------------+
              |  label_for_score()    |
              |  attribution + label  |
              |  text (asymmetric     |
              |  thresholds)          |
              +-----------------------+
                         |
                         | content_id, attribution,
                         | confidence, label_text,
                         | both signal scores
                         v
              +-----------------------+
              |  storage.py           |
              |  SQLite audit log row |
              +-----------------------+
                         |
                         v
              JSON response to caller
       { content_id, attribution, confidence, label, signals }


                          APPEAL FLOW
                          ============

   creator
      |
      | POST /appeal { content_id, creator_reasoning }
      v
   +----------------+
   | Flask route    |
   |   /appeal      |
   +----------------+
      |
      | look up existing row by content_id
      v
   +-----------------------+
   | storage.py            |
   | status -> under_review|
   | log appeal_reasoning  |
   | + appeal_timestamp    |
   +-----------------------+
      |
      v
   JSON confirmation { content_id, status: under_review, message }
```

## 1. Detection Signals

- **Signal 1 — LLM classification (Groq, `llama-3.3-70b-versatile`).**
  Output: a JSON object `{ai_probability: float 0-1, reasoning: str}`.
  Measures holistic semantic/stylistic coherence — does this read like a
  plausible human voice, or like generic/synthesized phrasing? Captures
  things structural metrics can't (logical framing, hedging language,
  genericness of content) but can be fooled by AI text explicitly prompted
  to sound casual, and can misjudge unusual-but-genuine human voices
  (non-native speakers, translated text) as "off."

- **Signal 2 — Stylometric heuristics (pure Python, no external libs).**
  Output: a float 0–1 (`stylometric_ai_probability`), built from three
  equally-weighted sub-scores: sentence-length uniformity (coefficient of
  variation), vocabulary diversity (type-token ratio), and formal
  punctuation density vs. informal markers (`!`, `...`). Measures surface
  structural regularity. Captures a genuinely different property than
  Signal 1 (structural vs. semantic), but can't capture meaning at all and
  false-positives on disciplined, formal human writing (academic/legal
  style) — a known edge case (see below).

- **Signal 3 — Trained ML classifier (TF-IDF + Logistic Regression, stretch:
  Ensemble detection).** Output: a float 0–1. Trained offline
  (`training/train.py`) on the AI-GA dataset (28,662 balanced samples of
  human vs. GPT-3-generated scientific abstracts, sourced from GitHub since
  Hugging Face Hub isn't reachable from the training environment used here —
  see `training/download_data.py`). Measures learned lexical patterns via
  TF-IDF n-grams. Achieves 99.4% test accuracy **on its training domain**,
  but a feature inspection (`training/analyze_domain_shift.py`) shows part
  of that accuracy comes from genuine AI-framing tells ("this paper
  examines/presents/investigates...") and part comes from formatting
  artifacts specific to structured scientific abstracts — a real,
  measured generalization gap to this platform's actual domain (short
  creative/casual text), documented in full in `model_artifacts/eval_report.md`
  and README.md "Known Limitations."

**Combination:** weighted average, `0.65 * llm_score + 0.25 * stylo_score +
0.10 * trained_ml_score`. The LLM signal keeps the largest weight (broadest,
least domain-dependent coverage of AI tells). The trained ML signal gets
the smallest weight — not because it's a weak classifier in the abstract
(99.4% test accuracy says otherwise), but because that accuracy doesn't
transfer to this platform's domain, which was verified empirically rather
than assumed. An initial 0.15 weight was tested and rejected because it
caused a domain-mismatched signal to drag a clearly-AI reference case down
into "uncertain"; 0.10 was chosen specifically to fix that regression.

## 2. Uncertainty Representation

The combined score is a continuous 0–1 value, not a binary flag. It is
mapped to one of three attributions using **asymmetric thresholds**:

| Combined score        | Attribution     |
|------------------------|-----------------|
| `>= 0.70`              | `likely_ai`     |
| `<= 0.30`               | `likely_human`  |
| `0.30 < score < 0.70`  | `uncertain`     |

These are intentionally asymmetric around 0.5: it takes a *higher* score to
confidently call something AI than it takes a *low* score to confidently
call something human, and there's a wide 0.30–0.70 "uncertain" band in
between. This directly reflects the design decision that a false positive
(mislabeling a human's work as AI) is worse than a false negative on a
creative-writing platform — see the Hints section of the spec. A score of
0.6 does **not** get treated the same as 0.95; only 0.95 crosses into
`likely_ai`, and the displayed confidence percentage in the label text
scales with how far the score sits from the nearest boundary.

## 3. Transparency Label Design

Three fixed templates, filled in with the actual confidence percentage:

- **`likely_ai`:** "This content shows strong signals of AI generation. Our
  system detected writing patterns consistent with AI-generated text
  (confidence: {pct}%). If you believe this is incorrect, you can appeal
  this classification."
- **`likely_human`:** "This content shows strong signals of human
  authorship. Our system found writing patterns consistent with
  human-generated text (confidence: {pct}%)."
- **`uncertain`:** "We're not fully confident about this content's origin.
  Our system detected mixed signals that don't clearly match typical AI or
  human writing patterns (confidence: {pct}%). This content has not been
  conclusively classified, and no strong claim is being made either way."

The `likely_ai` label is the only one that proactively mentions the appeal
option in-line, since that's the case where a wrongly-labeled human creator
is most harmed and most needs to know their options.

## 4. Appeals Workflow

Any creator whose content has been classified can submit an appeal via
`POST /appeal` with `content_id` and `creator_reasoning` (free text
explaining why they believe the classification is wrong). The system:
looks up the submission by `content_id`, sets its `status` to
`under_review`, and stores the `creator_reasoning` and an
`appeal_timestamp` on the same row (so the original decision and the
appeal live together in the audit log — nothing is overwritten). No
automated re-classification happens. A human reviewer opening the appeal
queue (i.e., querying `GET /log` and filtering for `status ==
"under_review"`) would see: the original text, the original attribution
and confidence, both individual signal scores, and the creator's stated
reasoning, all in one row.

## 5. Anticipated Edge Cases

1. **Formal, disciplined human writing (academic, legal, technical).**
   The stylometric signal's sentence-length-uniformity and punctuation
   sub-scores are likely to score this as AI-like, since disciplined human
   prose is structurally regular in the same way AI text tends to be. The
   LLM signal is the main defense here, but it isn't perfect either — see
   the "borderline_formal_human" test case in Milestone 4, which the
   pipeline correctly resolves to `uncertain` rather than `likely_ai`,
   which is the desired behavior given the false-positive asymmetry.

2. **Non-native English speaker writing, especially more formal or
   hedge-heavy phrasing.** Both signals can misread careful, slightly
   stilted phrasing (common in second-language writing) as AI-like: the
   LLM signal because "hedging language" is one of its stated cues, and
   the stylometric signal because careful writers often produce more
   uniform sentence structure. This is a real fairness concern, and it's
   part of why the appeal workflow captures free-text reasoning rather
   than just a checkbox — a non-native speaker can explain their situation
   directly. The sample appeal request in Milestone 5 uses exactly this
   scenario.

## AI Tool Plan

- **M3 (submission endpoint + first signal):** Provide the "Detection
  Signals" section above and the architecture diagram to the AI tool. Ask
  it to generate the Flask app skeleton with a `POST /submit` stub, and the
  Groq-based `llm_signal()` function matching the described input/output
  contract. Verify by calling `llm_signal()` directly on 2–3 test strings
  and checking the returned `ai_probability` is a sane float before wiring
  into the route.

- **M4 (second signal + confidence scoring):** Provide "Detection Signals"
  + "Uncertainty Representation" + the diagram. Ask for the stylometric
  signal function and the `scoring.py` combination/threshold logic. Verify
  by running the four milestone-4 test inputs (clearly AI, clearly human,
  two borderline cases) and confirming scores separate in the expected
  direction and the borderline cases land in `uncertain`, not a confident
  bucket.

- **M5 (production layer):** Provide "Transparency Label Design" +
  "Appeals Workflow" + the diagram. Ask for the label-generation function
  and the `POST /appeal` endpoint. Verify by requesting all three label
  variants directly (feed in low/mid/high confidence scores) and confirming
  text matches the spec exactly, then confirm an appeal call updates
  `status` to `under_review` in `GET /log`.

- **M6 (stretch — ensemble detection / trained ML signal):** Provide the
  "Signal 3" description above and the updated diagram. Ask for: the
  training script (TF-IDF + Logistic Regression on a labeled dataset) and
  the signal module that loads the saved model. Verify by (1) checking the
  in-domain evaluation report for sane, non-degenerate metrics, (2)
  inspecting top model features to check for dataset-specific artifacts
  rather than genuine signal, and (3) running the trained model on the
  same four milestone-4 reference cases used for the other two signals to
  check whether it actually transfers to the platform's real domain before
  trusting it in the ensemble. Re-tune the ensemble weight based on what
  step (3) shows, rather than assuming the in-domain accuracy number is
  the whole story.
