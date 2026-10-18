"""
app.py
------
Flask backend for the Handwritten Digit Recognition application.

Routes
------
GET  /                  – Serve the single-page application
GET  /api/overview      – Dataset stats + model configuration + metrics
GET  /api/evaluation    – Detailed evaluation metrics
GET  /api/dataset       – Class distribution + sample images
POST /api/predict       – Predict a user-drawn digit
"""

import os
import json
import logging
import numpy as np
from flask import Flask, jsonify, request, abort, send_from_directory

import model_utils

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Ensure model is loaded at startup ────────────────────────────────────────
_model_loaded = False

@app.before_request
def _load_model_once():
    """Trigger model load (or auto-train) on the very first request only."""
    global _model_loaded
    if not _model_loaded:
        model_utils.load_or_train_model()
        _model_loaded = True


# ─── Serve SPA ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# ─── API: Overview ────────────────────────────────────────────────────────────
@app.route("/api/overview")
def api_overview():
    meta = model_utils.get_metadata()
    return jsonify({
        "n_samples"       : meta["n_samples"],
        "n_classes"       : meta["n_classes"],
        "image_shape"     : meta["image_shape"],
        "intensity_range" : meta["intensity_range"],
        "n_train"         : meta["n_train"],
        "n_test"          : meta["n_test"],
        "model_name"      : meta["model_name"],
        "kernel"          : meta["kernel"],
        "C"               : meta["C"],
        "gamma"           : meta["gamma"],
        "accuracy"        : meta["accuracy"],
        "precision"       : meta["precision"],
        "recall"          : meta["recall"],
        "f1_score"        : meta["f1_score"],
    })


# ─── API: Evaluation ──────────────────────────────────────────────────────────
@app.route("/api/evaluation")
def api_evaluation():
    meta = model_utils.get_metadata()
    return jsonify({
        "accuracy"        : meta["accuracy"],
        "precision"       : meta["precision"],
        "recall"          : meta["recall"],
        "f1_score"        : meta["f1_score"],
        "confusion_matrix": meta["confusion_matrix"],
        "per_class"       : meta["per_class"],
    })


# ─── API: Dataset ─────────────────────────────────────────────────────────────
@app.route("/api/dataset")
def api_dataset():
    digits = model_utils.get_digits_dataset()
    meta   = model_utils.get_metadata()

    # Class distribution
    unique, counts = np.unique(digits.target, return_counts=True)
    class_dist = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    # Sample images – pick up to 5 per digit, store as 8x8 list-of-lists
    samples = []
    for digit_class in range(10):
        idx = np.where(digits.target == digit_class)[0][:5]
        for i in idx:
            img_8x8 = digits.images[i].tolist()   # 8 rows × 8 cols
            flat_64  = digits.data[i].tolist()     # flat 64 values
            samples.append({
                "label"   : int(digit_class),
                "image"   : img_8x8,
                "flat"    : flat_64,
            })

    return jsonify({
        "n_samples"       : meta["n_samples"],
        "n_classes"       : meta["n_classes"],
        "image_shape"     : meta["image_shape"],
        "intensity_range" : meta["intensity_range"],
        "class_distribution": class_dist,
        "samples"         : samples,
    })


# ─── API: Predict ─────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON (Content-Type: application/json)."}), 415

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Could not parse JSON body."}), 400

    # Mode 1: High-resolution canvas image (freehand drawing)
    if "image" in data and isinstance(data["image"], str):
        try:
            result = model_utils.predict_canvas_image(data["image"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.exception("Prediction failed on canvas image")
            return jsonify({"error": "Internal server error during image prediction."}), 500

        return jsonify({
            "predicted_digit"     : result["predicted_digit"],
            "decision_scores"     : result["decision_scores"],
            "preprocessed_pixels" : result.get("preprocessed_pixels", []),
        })

    # Mode 2: Pre-extracted 64-element pixel array (from preset examples)
    if "pixels" in data:
        raw_pixels = data["pixels"]
        if not isinstance(raw_pixels, list):
            return jsonify({"error": "'pixels' must be a JSON array."}), 400

        if len(raw_pixels) != 64:
            return jsonify({
                "error": f"Expected exactly 64 pixel values, received {len(raw_pixels)}."
            }), 422

        for i, v in enumerate(raw_pixels):
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                return jsonify({"error": f"Pixel at index {i} is not a valid number."}), 422

        try:
            result = model_utils.predict_digit(raw_pixels)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.exception("Unexpected error during prediction")
            return jsonify({"error": "Internal server error during prediction."}), 500

        return jsonify({
            "predicted_digit"     : result["predicted_digit"],
            "decision_scores"     : result["decision_scores"],
            "preprocessed_pixels" : result.get("preprocessed_pixels", []),
        })

    return jsonify({"error": "Missing 'image' or 'pixels' in request body."}), 400


# ─── Error handlers ───────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Handwritten Digit Recognition - Flask App")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
