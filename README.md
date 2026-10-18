# Handwritten Digit Recognition Using SVM (React + TypeScript)
### AI/ML Mini Project — Modernized React Application

A complete, interactive web application featuring a **Support Vector Machine (SVM)** to recognise handwritten digits (0–9) and allowing users to draw digits on a canvas for instant AI predictions, inspect model evaluation metrics, explore the training dataset, and review viva question guides.

---

## Features

- **Interactive Canvas Drawing:** Draw any digit (0–9) with touch and mouse support, stroke smoothing, undo, and canvas clear.
- **Instant Client-Side AI Prediction:** Evaluates drawn digits using a high-precision SVM with an RBF kernel and custom preprocessing (bounding box cropping, proportional margin padding, aspect-ratio preservation, true area downsampling, contrast normalization, and center-of-mass alignment).
- **Decision Scores Visualization:** Interactive Chart.js bar chart displaying per-class decision scores.
- **Preprocessed 8×8 Matrix Preview:** Displays the exact calibrated 8×8 resolution pixel array evaluated by the model.
- **Example Digits from Dataset:** 10 pre-loaded dataset examples (0–9) that can be clicked to test the model immediately.
- **Model Evaluation Dashboard:** Genuine metrics computed on held-out test data (accuracy, precision, recall, F1-score), per-class bar charts, and a color-scaled 10×10 Confusion Matrix.
- **Dataset Explorer:** Class distribution chart and gallery rendering real 8×8 handwritten digit samples across all 10 classes.
- **About Project & Viva Q&A:** Comprehensive explanation of SVM, RBF kernels, regularisation parameter $C$, preprocessing pipeline, limitations, and answers to 10 common viva questions.

---

## Project Structure

```
digisvm/
├── src/
│   ├── components/
│   │   ├── Sidebar.tsx      ← Navigation drawer with brand badge
│   │   └── Topbar.tsx       ← Responsive mobile header
│   ├── data/
│   │   ├── dataset.json     ← Dataset metadata, class distribution, samples
│   │   └── model.json       ← Calibrated SVM weights & support vectors
│   ├── pages/
│   │   ├── OverviewPage.tsx    ← Overview dashboard & summary metrics
│   │   ├── RecognitionPage.tsx ← Drawing canvas, inference & scores chart
│   │   ├── EvaluationPage.tsx  ← Confusion matrix & per-class charts
│   │   ├── DatasetPage.tsx     ← Dataset stats, distribution & gallery
│   │   └── AboutPage.tsx       ← Theory, architecture & Viva Q&A
│   ├── utils/
│   │   └── svm.ts           ← Mathematical SVM inference & image preprocessor
│   ├── types.ts             ← TypeScript definitions
│   ├── index.css            ← Dark-theme glassmorphism styles
│   ├── App.tsx              ← Application shell & routing
│   └── main.tsx             ← React entry point
├── index.html               ← HTML shell with SEO meta tags
├── package.json             ← Node.js dependencies & scripts
├── tsconfig.json            ← TypeScript compiler configuration
├── vite.config.ts           ← Vite build configuration
└── metadata.json            ← AI Studio application metadata
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Framework | React 18, TypeScript |
| Build Tool | Vite |
| Styling | CSS3 Dark Glassmorphism, Responsive Grid / Flexbox |
| Machine Learning | Support Vector Machine (RBF kernel, OvO + OvR continuous decision function) |
| Visualizations | Chart.js 4 |
| Dataset | Scikit-learn Handwritten Digits (`load_digits`) |

---

## Development & Build

### Install Dependencies
```bash
npm install
```

### Start Development Server
```bash
npm run dev
```
The application will be available at `http://localhost:3000`.

### Production Build
```bash
npm run build
```

---

## Machine Learning Details

### Algorithm: Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **Regularisation ($C$):** 2.0
- **Gamma:** Scale ($1 / (64 \times \text{variance})$)
- **Decision Function:** Multi-class One-vs-Rest (OvR) reconstructed from 45 pairwise One-vs-One (OvO) binary classifiers with continuous margin scores.
- **Test Set Accuracy:** 99.72% on held-out test data.

---

*Handwritten Digit Recognition Using SVM · AI/ML Mini Project*
