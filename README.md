# Handwritten Digit Recognition Using SVM
### AI/ML Mini Project — College Submission

A complete, interactive web application that trains a **Support Vector Machine (SVM)** to recognise handwritten digits (0–9) and lets users draw digits in a browser canvas for real-time AI predictions.

---

## Project Structure

```
handwritten-digit-recognition/
├── app.py               ← Flask backend + all API routes
├── train_model.py       ← Dataset loading, SVM training, saving model
├── model_utils.py       ← Model loading, image preprocessing, prediction
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── models/
│   └── svm_pipeline.joblib   ← Saved model (auto-generated)
├── templates/
│   └── index.html       ← Single-page application (HTML)
└── static/
    ├── css/
    │   └── style.css    ← All styling (dark theme, glassmorphism)
    └── js/
        └── script.js    ← Canvas drawing, API calls, Chart.js charts
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Machine Learning | Scikit-learn (SVM with RBF kernel) |
| Backend | Python, Flask |
| Preprocessing | NumPy, StandardScaler, Scikit-learn Pipeline |
| Model Persistence | Joblib |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js (CDN) |
| Dataset | sklearn.datasets.load_digits() |

> **Internet required** only for loading Chart.js from CDN. All ML code runs locally.

---

## Windows Setup (Step by Step)

### Prerequisites
- Python 3.9 or newer: https://python.org/downloads
- VS Code (recommended): https://code.visualstudio.com

### 1. Open the project folder in terminal

```
cd "C:\path\to\aiml mp"
```

Or right-click the folder in File Explorer → "Open in Terminal"

### 2. Create a virtual environment

```
python -m venv venv
```

### 3. Activate the virtual environment

**Command Prompt (recommended):**
```
venv\Scripts\activate.bat
```

**PowerShell** (if allowed):
```
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks activation with a security error, use **Command Prompt** instead
> (search for "cmd" in Start menu). You do NOT need to change security policies.

You will see `(venv)` at the start of the prompt when activated.

### 4. Upgrade pip and install dependencies

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Train the SVM model

```
python train_model.py
```

This will:
- Load the Scikit-learn digits dataset (1,797 images)
- Split 80% for training, 20% for testing
- Train the SVM pipeline (StandardScaler + SVC with RBF kernel)
- Print accuracy, precision, recall, F1-score, and confusion matrix
- Save the model to `models/svm_pipeline.joblib`

Expected output:
```
[5/5] Evaluating on test set ...
      Accuracy  : ~98–99%
      Precision : ~98–99%  (macro)
      Recall    : ~98–99%  (macro)
      F1-score  : ~98–99%  (macro)
```

### 6. Start the web application

```
python app.py
```

Expected output:
```
============================================================
  Handwritten Digit Recognition - Flask App
  Open http://127.0.0.1:5000 in your browser
  Press Ctrl+C to stop the server
============================================================
```

**Keep this terminal window open** while the app is running.

### 7. Open the application

Open your browser and go to:
```
http://127.0.0.1:5000
```

### 8. Stop the server

Press `Ctrl+C` in the terminal.

---

## How to Use the Application

1. **Navigate** using the left sidebar
2. **Digit Recognition** → Draw a single digit (0–9) on the white canvas
3. Click **Predict Digit** → the SVM model returns its prediction instantly
4. View the **SVM Decision Scores** bar chart (one score per digit class)
5. Click an **example digit** from the dataset gallery to test recognition on a known image
6. Visit **Model Evaluation** to see accuracy, confusion matrix, and per-class metrics
7. Visit **Dataset Explorer** to browse the training dataset samples
8. Visit **About Project** for explanations and viva Q&A

---

## How to Retrain the Model

If you want to retrain (e.g., to experiment with different hyperparameters):

```
python train_model.py
```

The old `models/svm_pipeline.joblib` will be overwritten. Restart Flask afterward:
```
python app.py
```

---

## Project Summary (for College Report)

### Problem Statement
Recognise handwritten digits (0–9) from image data using machine learning and serve predictions via an interactive web application.

### Objective
Train and evaluate an SVM classifier on the Scikit-learn digits dataset, then expose it through a Flask API consumed by a browser-based drawing canvas.

### Algorithm
**Support Vector Machine (SVM)** with **Radial Basis Function (RBF) kernel**
- `C=10` — regularisation parameter (soft margin)
- `gamma='scale'` — automatically sets gamma = 1/(n_features × variance)
- `decision_function_shape='ovr'` — One-vs-Rest for 10 classes

### Dataset
`sklearn.datasets.load_digits()`
- 1,797 grayscale images, 8×8 pixels, 64 features per image
- Pixel intensity: 0–16
- Classes: 0–9 (balanced)
- Split: 80% training (1,437 samples), 20% testing (360 samples)

### Preprocessing
1. Flatten 8×8 image to 64-feature vector
2. `StandardScaler` (fit on training data only) — zero mean, unit variance
3. For user-drawn digits: crop bounding box, resize to 8×8, invert (white→dark background), scale 0–255 → 0–16

### Evaluation Metrics
All metrics computed on the held-out test set:

| Metric | Value (typical) |
|---|---|
| Accuracy | ~98–99% |
| Precision (Macro) | ~98–99% |
| Recall (Macro) | ~98–99% |
| F1-Score (Macro) | ~98–99% |

### Applications
- Bank cheque digit reading
- Postal code recognition for mail sorting
- Government form digitisation
- Educational AI/ML demonstrations
- Document OCR pre-processing

### Limitations
- Model trained on tiny 8×8 images; browser strokes lose detail when downscaled
- Writing style differences between dataset authors and app users cause some errors
- SVM decision scores are NOT calibrated probabilities
- Larger CNN models (e.g., LeNet on MNIST) would perform far better on browser input

---

## Viva Questions and Answers

**Q1: What is a Support Vector Machine?**
A supervised ML algorithm that finds the optimal hyperplane maximising the margin between classes in feature space. Points closest to the boundary are called support vectors.

**Q2: What is the RBF kernel?**
The Radial Basis Function kernel computes similarity using a Gaussian function, mapping data to higher-dimensional space where it becomes linearly separable. It handles non-linear patterns like digit strokes.

**Q3: What does parameter C control?**
C is the regularisation parameter. Higher C → smaller margin, fewer training errors (risks overfitting). C=10 gives moderate regularisation suitable for this dataset.

**Q4: What does gamma='scale' mean?**
gamma is automatically set to 1/(n_features × X.var()) = 1/(64 × variance). This adapts the kernel width to the actual data spread.

**Q5: Why use StandardScaler?**
SVM with RBF kernel is distance-based. Unscaled features dominate distance calculations. StandardScaler ensures each pixel feature has equal influence.

**Q6: What is a Pipeline and why use it?**
A Scikit-learn Pipeline chains preprocessing and model steps. The scaler is fitted only on training data and automatically applied during prediction, preventing data leakage.

**Q7: What is the train-test split?**
80% of data (1,437 images) trains the model. 20% (360 images) is held out for evaluation. This gives an honest performance estimate on unseen data.

**Q8: What does the confusion matrix show?**
A 10×10 table. Entry [i][j] = count of actual digit-i samples predicted as digit j. The diagonal shows correct predictions; off-diagonal entries show errors.

**Q9: What is macro averaging?**
Computes the metric independently for each of the 10 classes, then takes the simple average across all classes (treating each class equally regardless of sample count).

**Q10: Why are SVM scores not probabilities?**
SVM decision scores = signed distance from the decision boundary. They indicate which class the model favours, but are not calibrated probabilities. Platt scaling can convert them, but requires additional training.

---

## API Reference

| Method | Route | Description |
|---|---|---|
| GET | `/api/overview` | Dataset stats + model config + summary metrics |
| GET | `/api/evaluation` | Full metrics: accuracy, CM, per-class P/R/F1 |
| GET | `/api/dataset` | Class distribution + sample images |
| POST | `/api/predict` | Predict digit from 64 pixel values (JSON) |

### POST /api/predict — Request Body
```json
{
  "pixels": [0, 5, 12, 16, 8, 0, 0, 0, ...]
}
```
- Array of exactly 64 float values
- Each value in range [0, 16]

### POST /api/predict — Success Response
```json
{
  "predicted_digit": 7,
  "decision_scores": [-0.45, -1.2, 0.3, 0.8, -0.9, -1.1, -0.7, 2.3, -0.5, -0.6]
}
```

---

*Built with Python, Flask, Scikit-learn, and Vanilla JavaScript*
*College Mini Project — Handwritten Digit Recognition Using SVM*
