from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import storage
from scoring import score_submission
from signals_llm import llm_signal
from signals_stylometric import stylometric_signal
from signals_trained_ml import trained_ml_signal

app = Flask(__name__)

# Rate limiting: 10/minute, 100/day per client. See README "Rate Limiting"
# section for the reasoning behind these specific numbers.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

storage.init_db()


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    creator_id = data.get("creator_id", "unknown")

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400

    llm_result = llm_signal(text)
    stylo_result = stylometric_signal(text)
    trained_ml_result = trained_ml_signal(text)

    scored = score_submission(
        llm_ai_probability=llm_result["llm_ai_probability"],
        stylometric_ai_probability=stylo_result["stylometric_ai_probability"],
        trained_ml_ai_probability=trained_ml_result["trained_ml_ai_probability"],
    )

    content_id = storage.create_submission(
        creator_id=creator_id,
        text=text,
        llm_score=llm_result["llm_ai_probability"],
        llm_reasoning=llm_result.get("reasoning", ""),
        stylometric_score=stylo_result["stylometric_ai_probability"],
        trained_ml_score=trained_ml_result["trained_ml_ai_probability"],
        confidence=scored["confidence"],
        attribution=scored["attribution"],
        label_text=scored["label_text"],
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": scored["attribution"],
            "confidence": scored["confidence"],
            "label": scored["label_text"],
            "signals": {
                "llm_ai_probability": llm_result["llm_ai_probability"],
                "llm_reasoning": llm_result.get("reasoning", ""),
                "stylometric_ai_probability": stylo_result["stylometric_ai_probability"],
                "stylometric_components": stylo_result["components"],
                "trained_ml_ai_probability": trained_ml_result["trained_ml_ai_probability"],
            },
        }
    )


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id", "").strip()
    creator_reasoning = data.get("creator_reasoning", "").strip()

    if not content_id or not creator_reasoning:
        return (
            jsonify({"error": "Fields 'content_id' and 'creator_reasoning' are required."}),
            400,
        )

    submission = storage.get_submission(content_id)
    if submission is None:
        return jsonify({"error": f"No submission found for content_id '{content_id}'."}), 404

    storage.record_appeal(content_id, creator_reasoning)

    return jsonify(
        {
            "content_id": content_id,
            "status": "under_review",
            "message": "Appeal received and logged. A human reviewer will examine this classification.",
        }
    )


@app.route("/log", methods=["GET"])
def log():
    limit = request.args.get("limit", default=20, type=int)
    entries = storage.get_recent_log(limit=limit)
    return jsonify({"entries": entries})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
