"""
Metro-ASR HTTP Server.

Serves a Metro-ASR model via REST API for inference.
Supports requests via curl, Postman, or Python requests.

Usage:
    python scripts/serve.py

API Endpoints:
    POST /transcribe          — Transcribe audio file
    POST /transcribe/batch    — Transcribe multiple audio files
    GET  /health              — Server health check
    GET  /info                — Model info

Examples:
    # curl
    curl -X POST http://localhost:8000/transcribe \
        -F "audio=@audio.wav"

    # curl with beam search
    curl -X POST http://localhost:8000/transcribe \
        -F "audio=@audio.wav" \
        -F "beam_search=true"

    # Python
    import requests
    files = {"audio": open("audio.wav", "rb")}
    resp = requests.post("http://localhost:8000/transcribe", files=files)
    print(resp.json()["text"])
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils import enable_utf8_stdout

enable_utf8_stdout()

# ========================= CONFIGURATION =========================
# A local directory (config.yaml + model.pt + bpe.model), a size alias
# ("small"), or a HuggingFace repo id. Override with METRO_MODEL so this
# script works unmodified wherever there is no local checkpoints/ directory
# (a fresh clone, a hosted notebook, a container).
MODEL = os.environ.get("METRO_MODEL", "checkpoints")
DEVICE = os.environ.get("METRO_DEVICE", "cpu")

LM_PATH = os.environ.get("METRO_LM", "auto")   # "auto" = KenLM binary next to the checkpoint; empty = greedy only
BEAM_WIDTH = int(os.environ.get("METRO_BEAM_WIDTH", 100))
LM_ALPHA = float(os.environ.get("METRO_LM_ALPHA", 0.5))
LM_BETA = float(os.environ.get("METRO_LM_BETA", 5.0))

HOST = os.environ.get("METRO_HOST", "0.0.0.0")
PORT = int(os.environ.get("METRO_PORT", 8000))
# =================================================================

if not LM_PATH:
    LM_PATH = None

from flask import Flask, request, jsonify

from metro_asr.engine import MetroASREngine

app = Flask(__name__)
engine = None


def load_engine():
    global engine
    print(f"Loading Metro-ASR ({MODEL})...")

    engine = MetroASREngine.from_pretrained(
        MODEL,
        device=DEVICE,
        lm_path=LM_PATH,
        beam_width=BEAM_WIDTH,
        lm_alpha=LM_ALPHA,
        lm_beta=LM_BETA,
    )

    print(f"  Model loaded: {engine.param_count:,} params on {DEVICE}")
    print(f"  LM: {'loaded' if engine.has_lm else 'none'}")
    print(f"  Server: http://{HOST}:{PORT}")
    print()


@app.route("/transcribe", methods=["POST"])
def transcribe():
    beam_search = request.form.get("beam_search", "false").lower() == "true"

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use -F 'audio=@file.wav'"}), 400

    audio_file = request.files["audio"]
    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = engine.transcribe(tmp_path, beam_search=beam_search)
        return jsonify({
            "text": result.text,
            "duration": round(result.duration, 3),
            "rtf": round(result.rtf, 6),
            "inference_time": round(result.inference_time, 4),
            "decoding_time": round(result.decoding_time, 4),
            "method": result.method,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/transcribe/batch", methods=["POST"])
def transcribe_batch():
    beam_search = request.form.get("beam_search", "false").lower() == "true"
    audio_files = request.files.getlist("audio")

    if not audio_files:
        return jsonify({"error": "No audio files provided."}), 400

    tmp_paths = []
    try:
        for af in audio_files:
            suffix = os.path.splitext(af.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                af.save(tmp.name)
                tmp_paths.append(tmp.name)

        results = engine.transcribe_batch(tmp_paths, beam_search=beam_search)
        return jsonify({
            "results": [
                {
                    "text": r.text,
                    "duration": round(r.duration, 3),
                    "rtf": round(r.rtf, 6),
                    "method": r.method,
                }
                for r in results
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in tmp_paths:
            os.unlink(p)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model": MODEL,
        "params": engine.param_count,
        "device": str(engine.device),
        "sample_rate": engine.sample_rate,
        "lm_loaded": engine.has_lm,
    })


if __name__ == "__main__":
    load_engine()
    app.run(host=HOST, port=PORT, threaded=True)
