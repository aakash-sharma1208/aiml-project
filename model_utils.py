"""
model_utils.py
--------------
Utility functions for:
  - Loading / auto-training the SVM pipeline
  - Preprocessing browser-drawn digit images (full-canvas PIL area box filtering,
    proportional margin, square padding, contrast normalization, center of mass alignment)
  - Processing raw pixel arrays from preset examples
  - Returning predictions and calibrated decision scores
"""

import os
import io
import re
import base64
import logging
import numpy as np
import joblib
from PIL import Image, ImageFilter
from scipy.ndimage import center_of_mass, shift
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ─── Custom Pixel Scaler ───────────────────────────────────────────────────────
class PixelScaler(BaseEstimator, TransformerMixin):
    """
    Uniformly scales pixel intensities from [0, 16] to [0.0, 1.0].
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32) / 16.0


MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_pipeline.joblib")

# Module-level cache so we load from disk only once
_artifact = None


# ─── Model loading ────────────────────────────────────────────────────────────

def load_or_train_model():
    """Return the saved artifact dict. Train automatically if not present."""
    global _artifact
    if _artifact is not None:
        return _artifact

    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found – training now …")
        from train_model import train_and_save
        _artifact = train_and_save()
    else:
        logger.info("Loading saved model from %s", MODEL_PATH)
        _artifact = joblib.load(MODEL_PATH)

    return _artifact


def get_pipeline():
    return load_or_train_model()["pipeline"]


def get_metadata():
    return load_or_train_model()["metadata"]


def get_digits_dataset():
    return load_or_train_model()["digits"]


# ─── High-Precision Canvas Image Preprocessing ─────────────────────────────────

def process_pil_image(pil_img: Image.Image) -> np.ndarray:
    """
    Preprocess a 280x280 grayscale PIL Image of a drawn digit into a calibrated 8x8 input.

    Pipeline:
      1. Grayscale inversion: white background (255) -> 0, black ink (0) -> 255.
      2. Thresholding: Detect active drawn strokes.
      3. Adaptive stroke dilation: If user wrote with a thin/light stroke, dilate
         morphologically so strokes match the standard ~2px width in 8x8 resolution.
      4. Bounding box cropping with proportional margin (12%).
      5. Aspect-ratio preservation by padding into a square matrix.
      6. True area-box downsampling (Image.Resampling.BOX) to 8x8 pixels.
      7. Dynamic contrast stretching so peak drawn ink reaches 16.0.
      8. Center-of-Mass alignment to (3.5, 3.5), matching dataset distribution.
    """
    # 1. Convert to grayscale array and invert
    gray = pil_img.convert("L")
    arr = 255.0 - np.array(gray, dtype=float)

    # 2. Check for drawn ink
    mask = arr > 25.0
    if np.sum(mask) < 20:
        return None

    ymin, ymax = np.where(np.any(mask, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(mask, axis=0))[0][[0, -1]]

    h = ymax - ymin + 1
    w = xmax - xmin + 1

    if h < 10 and w < 10:
        return None

    # 3. Adaptive stroke dilation for light/thin strokes
    density = np.sum(mask) / (h + w)
    img_inv = Image.fromarray(arr.astype(np.uint8))
    if density < 14:
        img_inv = img_inv.filter(ImageFilter.MaxFilter(5))
    elif density < 20:
        img_inv = img_inv.filter(ImageFilter.MaxFilter(3))

    arr_d = np.array(img_inv, dtype=float)
    mask_d = arr_d > 25.0
    ymin, ymax = np.where(np.any(mask_d, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(mask_d, axis=0))[0][[0, -1]]
    h = ymax - ymin + 1
    w = xmax - xmin + 1

    # 4. Proportional padding (12%)
    pad = max(10, int(round(max(h, w) * 0.12)))
    orig_h, orig_w = arr_d.shape
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(orig_h - 1, ymax + pad)
    xmax = min(orig_w - 1, xmax + pad)

    cropped = arr_d[ymin:ymax + 1, xmin:xmax + 1]
    ch, cw = cropped.shape

    # 5. Square padding to preserve aspect ratio
    max_side = max(ch, cw)
    square = np.zeros((max_side, max_side), dtype=float)
    sy = (max_side - ch) // 2
    sx = (max_side - cw) // 2
    square[sy:sy + ch, sx:sx + cw] = cropped

    # 6. True box-averaging downsampling to 8x8
    sq_img = Image.fromarray(square.astype(np.uint8))
    res8 = sq_img.resize((8, 8), Image.Resampling.BOX)
    arr8 = np.array(res8, dtype=float)

    # 7. Dynamic contrast stretching (scale peak ink to 16.0)
    if arr8.max() > 0:
        arr8 = (arr8 / arr8.max()) * 16.0

    # 8. Center-of-Mass alignment to (3.5, 3.5)
    cy, cx = center_of_mass(arr8)
    if not (np.isnan(cy) or np.isnan(cx)):
        dy = float(np.clip(3.5 - cy, -1.2, 1.2))
        dx = float(np.clip(3.5 - cx, -1.2, 1.2))
        arr8 = shift(arr8, [dy, dx], mode="constant", cval=0.0)
    arr8 = np.clip(arr8, 0.0, 16.0)
    return arr8.flatten().astype(np.float32)


def process_canvas_data_url(data_url: str) -> np.ndarray:
    """Decode a base64 PNG data URL from the browser canvas and process it."""
    # Strip data URL header if present
    match = re.search(r"base64,(.*)", data_url)
    if match:
        raw_b64 = match.group(1)
    else:
        raw_b64 = data_url

    try:
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        raise ValueError(f"Invalid image data: {exc}")

    processed = process_pil_image(pil_img)
    if processed is None:
        raise ValueError("The canvas appears to be blank. Please draw a digit first.")

    return processed.reshape(1, -1)


# ─── Raw Pixel Preprocessing (for preset example samples) ──────────────────────

def preprocess_canvas_pixels(raw_pixels: list) -> np.ndarray:
    """Preprocess a flat list of 64 pixels sent from preset example buttons."""
    if len(raw_pixels) != 64:
        raise ValueError(f"Expected 64 pixel values, got {len(raw_pixels)}.")

    arr = np.array(raw_pixels, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Pixel values must be finite numbers.")

    if arr.max() < 0.5:
        raise ValueError("The canvas appears to be blank. Please draw a digit first.")

    arr = np.clip(arr, 0.0, 16.0)
    return arr.reshape(1, -1)


# ─── Prediction Handlers ──────────────────────────────────────────────────────

def predict_from_features(X: np.ndarray) -> dict:
    """Run pipeline prediction and return scores and formatted output."""
    pipeline = get_pipeline()
    predicted = int(pipeline.predict(X)[0])

    raw_scores = pipeline.decision_function(X)[0]
    decision_scores = [round(float(s), 4) for s in raw_scores]
    preprocessed = [round(float(v), 2) for v in X[0]]

    return {
        "predicted_digit"     : predicted,
        "decision_scores"     : decision_scores,
        "preprocessed_pixels" : preprocessed,
    }


def predict_canvas_image(data_url: str) -> dict:
    """Predict digit from full canvas PNG data URL."""
    X = process_canvas_data_url(data_url)
    return predict_from_features(X)


def predict_digit(raw_pixels: list) -> dict:
    """Predict digit from 64-element pixel array."""
    X = preprocess_canvas_pixels(raw_pixels)
    return predict_from_features(X)
