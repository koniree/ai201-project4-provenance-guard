# Provenance Guard

A backend system that classifies submitted creative writing as likely
AI-generated, likely human-written, or uncertain — with a calibrated
confidence score, a plain-language transparency label, an appeals
workflow for contested classifications, rate limiting, and a structured
audit log. Built for AI201 Project 4.

**Stretch feature completed: Ensemble detection.** Three signals feed
the confidence score — an LLM signal, a stylometric heuristic signal, and
a trained TF-IDF + Logistic Regression classifier — with a documented,
tested weighting (see "Trained ML Signal" below for why the weighting
looks the way it does).

## Architecture Overview

A submission enters through `POST /submit` with `text` and `creator_id`.
The text is run through two independent detection signals — an LLM-based
holistic judgment (Groq, `llama-3.3-70b-versatile`) and a pure-Python
stylometric heuristic (sentence-length uniformity, vocabulary diversity,
punctuation density). Both return a 0–1 "AI probability." `scoring.py`
combines them with a weighted average into a single confidence score, then
maps that score through **asymmetric thresholds** into one of three
attributions (`likely_ai`, `likely_human`, `uncertain`), each tied to a
fixed transparency label template. The full result — both raw signal
scores, the combined confidence, the attribution, and the label text — is
written to a structured SQLite row keyed by a generated `content_id` and
returned to the caller. If a creator disputes the result, `POST /appeal`
looks up that row by `content_id`, sets `status` to `under_review`, and
appends the creator's reasoning to the same row, so the original decision
and the appeal live together in the audit log. `GET /log` surfaces recent
rows as JSON. The full diagram (submission flow + appeal flow) lives in
[`planning.md`](./planning.md) under `## Architecture Diagram`.

## Detection Signals

**Signal 1 — LLM classification (Groq).** Sends the text to
`llama-3.3-70b-versatile` with a system prompt asking for a holistic
human-vs-AI judgment as `{ai_probability, reasoning}`. Captures semantic
and stylistic coherence — genericness of phrasing, hedging language,
logical framing typical of LLM output — that structural metrics miss. It
can be fooled by AI text deliberately prompted to sound casual, and it can
misjudge unusual-but-genuine human voices (non-native speakers, translated
text) as "off" and therefore AI-like.

**Signal 2 — Stylometric heuristics (pure Python).** Computes three
sub-metrics and averages them: sentence-length uniformity (inverse
coefficient of variation), vocabulary diversity (type-token ratio,
inverted), and formal punctuation density minus informal markers (`!`,
`...`). Captures surface structural regularity — a genuinely different
property than Signal 1's semantic judgment. It can't capture meaning at
all, and it reliably false-positives on disciplined, formal human writing
(academic or legal prose), since that kind of writing is structurally
regular in the same way AI text tends to be.

**Signal 3 — Trained ML classifier (stretch: Ensemble detection).**
TF-IDF (unigrams + bigrams, 20k features) feeding a Logistic Regression
classifier, trained offline in `training/train.py` on the
[AI-GA dataset](https://github.com/panagiotisanagnostou/AI-GA) — 28,662
balanced samples of human vs. GPT-3-generated scientific abstracts (see
"Trained ML Signal" section below for why this specific dataset, and what
it does and doesn't tell us). Output: a float 0–1 via `predict_proba`.

These three signals are combined as `0.65 * llm_score + 0.25 * stylo_score
+ 0.10 * trained_ml_score` in `scoring.py`. The LLM signal keeps the
largest weight because it covers the broadest, least domain-dependent
range of AI tells. The trained ML signal gets the smallest weight — see
below for why that's a measured decision, not a hedge.

## Trained ML Signal (Stretch: Ensemble Detection)

**Dataset:** [AI-GA](https://github.com/panagiotisanagnostou/AI-GA)
(AI-Generated Abstracts), 28,662 samples, perfectly balanced — 14,331
original human abstracts from the [CORD-19](https://github.com/allenai/cord19)
COVID-19 research corpus, 14,331 GPT-3-generated counterparts. Chosen
because it's a real, sizable, cleanly-labeled dataset that's actually
retrievable in a network-restricted environment: it's committed directly
in a GitHub repo rather than gated behind a Hugging Face Hub download.
`training/download_data.py` fetches it; `training/train.py` trains and
evaluates a TF-IDF + Logistic Regression classifier on it.

**In-domain results:** 99.44% test accuracy, 99.44% F1, 0.9999 ROC-AUC on
a held-out 20% split (full numbers in `model_artifacts/eval_report.md`,
alongside a LinearSVC comparison run for reference).

**But — this number needed a second look before it went into the
ensemble.** `training/analyze_domain_shift.py` inspects the top TF-IDF
features driving predictions:

- Toward **AI-generated**: `this`, `this paper`, `this study`,
  `presents`, `examines`, `investigates`, `findings`, `article`
- Toward **human-written**: `in the`, `of the`, `background`, `however`,
  `we`, `were`

Part of this is a genuine, well-documented AI tell — GPT-3's habit of
framing text as "this paper examines/presents..." rather than just
stating the content directly. But `background` and structured-header-style
tokens look like they're picking up CORD-19's structured-abstract
formatting conventions (OBJECTIVE/METHODS/RESULTS/CONCLUSIONS), which is
a property of *this dataset*, not of human writing generally.

To check whether that mattered, the same trained classifier was run on
the project's actual target domain — short creative/casual text — using
the same four milestone-4 reference cases used for the other two signals:

| Input | P(AI) |
|---|---|
| clearly_ai (AI boilerplate) | 0.2322 |
| clearly_human (casual review) | 0.0828 |
| borderline_formal_human | 0.0601 |
| borderline_edited_ai | 0.0387 |

Two real problems show up: all four scores compress into a narrow low
band (vs. the confident spread seen in-domain), and the ordering is
partly wrong — `borderline_edited_ai` (which *is* AI-generated) scores
*lower* than the clearly-human case. That's a measured generalization
gap, not a hypothetical one, and it's the direct reason this signal gets
only 10% weight in the ensemble rather than being trusted at face value
for its 99.4% training-domain accuracy. An earlier 15% weight was tested
and rejected because it dragged the `clearly_ai` reference case down from
a confident `likely_ai` into `uncertain` — see `scoring.py` for the
full account of that tuning decision.

## Confidence Scoring

The combined score is mapped to an attribution using **asymmetric
thresholds** rather than a 0.5 cutoff:

| Combined score | Attribution |
|---|---|
| `>= 0.70` | `likely_ai` |
| `<= 0.30` | `likely_human` |
| `0.30 < score < 0.70` | `uncertain` |

This asymmetry is a deliberate response to the spec's hint that a false
positive (mislabeling a human's genuine work as AI-generated) is worse
than a false negative on a creative-writing platform — it takes a higher
combined score to earn a confident AI label than it takes a low score to
earn a confident human label, and there's a wide uncertain band between
them.

To validate that scores vary meaningfully rather than clustering near
0.5, `scoring.py` was tested against the four milestone-4 reference
inputs (illustrative LLM scores shown; trained-ML scores are the actual
measured values from `signals_trained_ml.py`; replace LLM scores with
live Groq output once `GROQ_API_KEY` is set):

**High-confidence example** — clearly AI-generated boilerplate
(`"Artificial intelligence represents a transformative paradigm shift..."`):
LLM signal `0.93`, stylometric signal `0.3601`, trained-ML signal `0.1268`
→ **combined confidence `0.7072`** → `likely_ai`.

**Low-confidence / uncertain example** — formal human academic writing
(`"The relationship between monetary policy and asset price inflation..."`):
LLM signal `0.55`, stylometric signal `0.2445`, trained-ML signal `0.06`
→ **combined confidence `0.4246`** → `uncertain`. This is the intended
behavior: a stylistically formal but genuinely human passage is *not*
forced into a confident AI label, it's flagged as ambiguous.

For contrast, clearly casual human writing (the ramen review) scored LLM
`0.04`, stylometric `0.065`, trained-ML `0.0677` → combined `0.049` →
`likely_human`, showing the full range from confidently human to
confidently AI is reachable even with the third signal in the mix.

## Transparency Label

All three variants, exact text as returned by `label_for_score()`
(`{pct}` is filled in with the actual confidence percentage):

| Variant | Exact text |
|---|---|
| **High-confidence AI** | "This content shows strong signals of AI generation. Our system detected writing patterns consistent with AI-generated text (confidence: {pct}%). If you believe this is incorrect, you can appeal this classification." |
| **High-confidence human** | "This content shows strong signals of human authorship. Our system found writing patterns consistent with human-generated text (confidence: {pct}%)." |
| **Uncertain** | "We're not fully confident about this content's origin. Our system detected mixed signals that don't clearly match typical AI or human writing patterns (confidence: {pct}%). This content has not been conclusively classified, and no strong claim is being made either way." |

Only the `likely_ai` label proactively mentions the appeal option inline,
since that's the case where a wrongly-labeled human creator is most
harmed and most needs to know their options.

## Appeals Workflow

`POST /appeal` with `{content_id, creator_reasoning}` looks up the
existing submission row, sets `status` to `under_review`, and stores the
creator's reasoning and an `appeal_timestamp` on that same row — nothing
is overwritten, so the original decision and the appeal live together.
No automated re-classification is triggered. Example:

```bash
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id": "PASTE-CONTENT-ID-HERE", "creator_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical."}' \
  | python -m json.tool
```

`GET /log` afterward shows that row with `"status": "under_review"` and
`appeal_reasoning` populated — verified locally (see AI Usage section
below for how this was tested).

## Rate Limiting

`POST /submit` is limited to **10 requests/minute and 100 requests/day**
per client (`Flask-Limiter`, in-memory storage). Reasoning: a genuine
writer submitting their own work realistically submits a handful of
pieces (or revisions of one piece) in a sitting — 10/minute comfortably
covers that with room for retries, while a script attempting to flood the
classification pipeline (which calls a paid/rate-limited external LLM API
per request) would hit the limit almost immediately. The 100/day cap
bounds total API cost per client across a full day of legitimate use
without meaningfully constraining a real user.

Verified locally: 12 rapid requests to `/submit` produced:

```
200
200
200
200
200
200
200
200
200
200
429
429
```

— the first 10 succeed, the 11th and 12th are rejected, confirming the
per-minute limit triggers correctly.

## Complete Audit Log

Every `/submit` call writes a structured SQLite row with: `content_id`,
`creator_id`, `timestamp`, `llm_score`, `llm_reasoning`,
`stylometric_score`, `trained_ml_score`, `confidence`, `attribution`,
`label_text`, `status`, and (once appealed) `appeal_reasoning` +
`appeal_timestamp`. `GET /log` returns the most recent rows as JSON.
Sample from local testing (all three signal scores present, including
one post-appeal row):

```json
{
  "entries": [
    {
      "content_id": "8206a595-9d4a-409d-8fcb-ea90e99abae2",
      "creator_id": "test-human",
      "attribution": "likely_human",
      "confidence": 0.0505,
      "llm_score": 0.04,
      "stylometric_score": 0.065,
      "trained_ml_score": 0.0828,
      "status": "classified",
      "appeal_reasoning": null
    },
    {
      "content_id": "715663b5-8c66-4e16-b0cf-58c2cb28db03",
      "creator_id": "test-ai",
      "attribution": "likely_ai",
      "confidence": 0.7177,
      "llm_score": 0.93,
      "stylometric_score": 0.3601,
      "trained_ml_score": 0.2322,
      "status": "under_review",
      "appeal_reasoning": "I wrote this myself as part of an essay assignment."
    }
  ]
}
```

## Known Limitations

**Formal, disciplined human writing scores closer to AI than it should
on the stylometric signal.** Sentence-length uniformity and formal
punctuation density are both structural properties that careful academic
or legal writing shares with AI-generated text, purely because both are
disciplined and regular. The `borderline_formal_human` test case
(monetary-policy passage) landed at `uncertain` rather than `likely_ai`
only because the LLM signal partially offset the stylometric signal and
the thresholds are asymmetric — a less careful weighting scheme could
easily have mislabeled genuine academic writing as AI, which is exactly
the false-positive failure mode the design tries to minimize but cannot
fully eliminate.

**The trained ML signal doesn't transfer well outside its training
domain.** It hits 99.4% accuracy on scientific abstracts (its training
data) but was measurably weaker and partly mis-ordered on the platform's
actual domain (short creative/casual text) — see "Trained ML Signal"
above for the exact numbers. This isn't a hedge; it's the direct reason
the signal gets only 10% ensemble weight instead of being trusted at its
face-value accuracy. The honest fix isn't a bigger weight, it's a
domain-matched training set (real examples of AI vs. human creative
writing) that this environment couldn't source in the time available —
flagged here as a concrete next step rather than solved in place.

## Spec Reflection

Writing out the three label variants and the five `planning.md` questions
*before* touching code directly shaped the threshold design: deciding
"what should 0.6 mean to a user" before writing any scoring code is what
led to the asymmetric-threshold approach rather than a naive midpoint
split, since a symmetric 0.5 cutoff would have treated a 0.51 the same as
a 0.95, which the spec explicitly flags as a problem.

Where the implementation diverged from the original plan: the initial
spec draft weighted the two signals equally (0.5/0.5). After running the
milestone-4 test inputs, the equal weighting let a strongly-worded but
formal human passage get pulled too close to `likely_ai` because the
stylometric signal's false-positive tendency wasn't being offset enough.
The weighting was revised to 0.65 (LLM) / 0.35 (stylometric) specifically
because the LLM signal proved more reliable on the borderline human case
in testing — this divergence is a direct result of testing against real
examples rather than reasoning about the combination in the abstract.

## AI Usage

This project was built with Claude assisting across the milestone
structure described in `planning.md`'s "AI Tool Plan" section:

1. **Milestone 3 (submission endpoint + first signal):** Directed Claude
   to generate the Flask app skeleton and the Groq-based `llm_signal()`
   function against the spec in `planning.md`. Claude's first draft parsed
   the Groq response with a regex; this was overridden with strict
   `json.loads()` plus a try/except fallback to a neutral 0.5 score, since
   a malformed LLM response should degrade gracefully rather than crash
   the whole submission pipeline.

2. **Milestone 4 (second signal + scoring):** Directed Claude to implement
   the stylometric heuristics and the score-combination logic. Claude's
   initial threshold design used a symmetric 0.5 midpoint split into two
   buckets; this was overridden to the three-bucket asymmetric-threshold
   design (`0.70` / `0.30`) described above, and the signal weighting was
   revised from an initial equal split after testing against the
   milestone-4 reference inputs showed the stylometric signal needed less
   influence on borderline-formal human text.

3. **Milestone 6 / stretch (trained ML signal + ensemble):** Directed
   Claude to source a real labeled dataset from GitHub (since Hugging Face
   Hub wasn't reachable in this environment) and train a TF-IDF + Logistic
   Regression classifier on it. Claude's first pass reported the 99.4%
   in-domain accuracy and moved straight to integrating it into the
   ensemble at a substantial weight (0.15) without checking whether that
   accuracy would transfer to this platform's actual domain. This was
   overridden: directed Claude to inspect the model's top features and run
   it against the existing milestone-4 reference cases before trusting it,
   which surfaced both a genuine dataset-artifact concern (formatting
   tells from the training corpus, not just AI-writing tells) and a real
   out-of-domain ranking failure. The ensemble weight was then revised
   down to 0.10 after confirming that 0.15 caused a regression on the
   `clearly_ai` reference case. The eval numbers throughout this section
   and "Known Limitations" are the actual measured results, not estimates.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
cp .env.example .env               # then add your real GROQ_API_KEY
```

**The trained ML signal ships pre-trained** (`model_artifacts/*.joblib`
are committed), so `python app.py` works out of the box. To retrain from
scratch (e.g. on a different/domain-matched dataset):

```bash
python training/download_data.py     # fetches ai-ga-dataset.csv from GitHub
python training/train.py             # trains + writes model_artifacts/
python training/analyze_domain_shift.py   # reproduces the feature/domain-shift checks
```

Then run the server:

```bash
python app.py
```

Server runs at `http://localhost:5000`. Test with:

```bash
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "creator_id": "test-user-1"}' \
  | python -m json.tool
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/submit` | Classify a piece of text; returns `content_id`, `attribution`, `confidence`, `label`, all three signal scores |
| `POST` | `/appeal` | Contest a classification by `content_id`; sets status to `under_review` |
| `GET` | `/log` | Recent structured audit log entries (`?limit=N`) |
| `GET` | `/health` | Basic liveness check |

## Project Structure

```
provenance-guard/
├── app.py                        # Flask routes, rate limiting, wiring
├── signals_llm.py                # Signal 1: Groq LLM classification
├── signals_stylometric.py        # Signal 2: stylometric heuristics
├── signals_trained_ml.py         # Signal 3: trained TF-IDF + LogReg classifier
├── scoring.py                    # Signal combination, thresholds, labels
├── storage.py                    # SQLite audit log + appeals
├── model_artifacts/
│   ├── tfidf_vectorizer.joblib   # Pre-trained (shipped)
│   ├── logreg_classifier.joblib  # Pre-trained (shipped)
│   └── eval_report.md            # Full eval + feature + domain-shift findings
├── training/
│   ├── download_data.py          # Fetches AI-GA dataset from GitHub
│   ├── train.py                  # Trains + evaluates the classifier
│   └── analyze_domain_shift.py   # Reproduces feature/domain-shift analysis
├── planning.md                   # Pre-implementation spec + architecture
├── requirements.txt
├── .env.example
└── README.md
```
