"""
train_model.py
--------------
Trains an SVM classifier on the Scikit-learn handwritten digits dataset
using standard, robust machine learning methodology:
  - 80/20 stratified train/test split (1,437 train / 360 held-out test).
  - Spatial shift augmentation (+/- 1 pixel) on training data for translation invariance.
  - Uniform pixel normalization (PixelScaler: [0, 16] -> [0.0, 1.0]).
  - SVM with RBF kernel (C=2.0, gamma='scale').
  - Full mathematical confusion matrix calculation with per-class True Positives (TP),
    False Positives (FP), False Negatives (FN), Support, Precision, Recall, and F1-Scores.
"""

import os
import joblib
import numpy as np
from scipy.ndimage import shift
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from model_utils import PixelScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_pipeline.joblib")


def train_and_save():
    """Train the SVM pipeline and compute full evaluation metrics."""
    print("=" * 65)
    print("  Handwritten Digit Recognition - High Precision SVM Trainer")
    print("=" * 65)

    # 1. Load dataset
    print("\n[1/4] Loading Scikit-learn digits dataset ...")
    digits = load_digits()
    X = digits.data          # shape (1797, 64)
    y = digits.target        # shape (1797,)

    print(f"      Total dataset samples : {len(X)}")
    print(f"      Classes               : {len(np.unique(y))}  ({np.unique(y).tolist()})")
    print(f"      Feature shape         : {X.shape[1]} features per image (8x8 pixels)")

    # 2. Stratified train/test split (80% train / 20% test, random_state=42)
    print("\n[2/4] Splitting data (80% train / 20% test, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Base training samples : {len(X_train)}")
    print(f"      Held-out test samples : {len(X_test)}")

    # Augment training set with spatial shifts to handle natural drawing variations
    print("      Applying translation shift augmentation to training set ...")
    X_train_aug = list(X_train)
    y_train_aug = list(y_train)
    for x_val, label in zip(X_train, y_train):
        img = x_val.reshape(8, 8)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            X_train_aug.append(shift(img, [dy, dx], mode="constant", cval=0.0).flatten())
            y_train_aug.append(label)

    X_train_aug = np.array(X_train_aug, dtype=np.float32)
    y_train_aug = np.array(y_train_aug)
    print(f"      Total augmented training samples: {len(X_train_aug)}")

    # 3. Build SVM pipeline
    print("\n[3/4] Building SVM pipeline (PixelScaler + SVC RBF, C=2.0, gamma='scale') ...")
    pipeline = Pipeline([
        ("scaler", PixelScaler()),
        ("svm",    SVC(kernel="rbf", C=2.0, gamma="scale", decision_function_shape="ovr"))
    ])

    pipeline.fit(X_train_aug, y_train_aug)
    print("      Training complete.")

    # 4. Genuine evaluation on held-out test set
    print("\n[4/4] Evaluating on held-out test set ...")
    y_pred = pipeline.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_test, y_pred,    average="macro", zero_division=0)
    f1    = f1_score(y_test, y_pred,        average="macro", zero_division=0)
    cm    = confusion_matrix(y_test, y_pred)
    errors = int(np.sum(y_test != y_pred))
    correct = len(y_test) - errors

    print(f"\n      Accuracy  : {acc*100:.2f}%  ({correct}/{len(y_test)} correct, {errors} errors)")
    print(f"      Precision : {prec*100:.2f}%  (macro)")
    print(f"      Recall    : {rec*100:.2f}%  (macro)")
    print(f"      F1-score  : {f1*100:.2f}%  (macro)")
    print("\n      Confusion Matrix:")
    print(cm)

    # Detailed per-class counts (TP, FP, FN, Support)
    tp_per_class      = [int(cm[i][i]) for i in range(10)]
    support_per_class = [int(np.sum(cm[i, :])) for i in range(10)]
    pred_total_class  = [int(np.sum(cm[:, i])) for i in range(10)]
    fn_per_class      = [int(support_per_class[i] - tp_per_class[i]) for i in range(10)]
    fp_per_class      = [int(pred_total_class[i] - tp_per_class[i]) for i in range(10)]

    per_class_prec = precision_score(y_test, y_pred, average=None, zero_division=0).tolist()
    per_class_rec  = recall_score(y_test, y_pred,    average=None, zero_division=0).tolist()
    per_class_f1   = f1_score(y_test, y_pred,        average=None, zero_division=0).tolist()

    metadata = {
        "n_samples"         : int(len(X)),
        "n_classes"         : int(len(np.unique(y))),
        "image_shape"       : [8, 8],
        "intensity_range"   : [int(X.min()), int(X.max())],
        "n_train"           : int(len(X_train)),
        "n_train_augmented" : int(len(X_train_aug)),
        "n_test"            : int(len(X_test)),
        "n_correct"         : correct,
        "n_errors"          : errors,
        "model_name"        : "SVM (RBF kernel + PixelScaler)",
        "kernel"            : "rbf",
        "C"                 : 2.0,
        "gamma"             : "scale",
        "accuracy"          : round(acc,  6),
        "precision"         : round(prec, 6),
        "recall"            : round(rec,  6),
        "f1_score"          : round(f1,   6),
        "confusion_matrix"  : cm.tolist(),
        "row_totals"        : support_per_class,
        "col_totals"        : pred_total_class,
        "per_class"         : {
            "precision" : [round(v, 6) for v in per_class_prec],
            "recall"    : [round(v, 6) for v in per_class_rec],
            "f1"        : [round(v, 6) for v in per_class_f1],
            "tp"        : tp_per_class,
            "fp"        : fp_per_class,
            "fn"        : fn_per_class,
            "support"   : support_per_class,
        },
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "pipeline" : pipeline,
        "metadata" : metadata,
        "X_train"  : X_train,
        "X_test"   : X_test,
        "y_train"  : y_train,
        "y_test"   : y_test,
        "digits"   : digits,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\n  Model saved to: {MODEL_PATH}")
    print("=" * 65)
    return artifact


if __name__ == "__main__":
    train_and_save()
