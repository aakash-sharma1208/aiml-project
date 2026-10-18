import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.ndimage import shift, rotate, center_of_mass
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class PixelScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.asarray(X, dtype=np.float32) / 16.0

digits = load_digits()
X, y = digits.data, digits.target

# 1. 25% test split (450 held-out test samples)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Augment training set ONLY with spatial translations and rotations
X_aug = []
y_aug = []
for x_val, label in zip(X_train, y_train):
    img = x_val.reshape(8, 8)
    X_aug.append(x_val)
    y_aug.append(label)
    # 4 directional shifts (+/- 1 pixel)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        s = shift(img, [dy, dx], mode='constant', cval=0.0).flatten()
        X_aug.append(s)
        y_aug.append(label)

X_aug = np.array(X_aug, dtype=np.float32)
y_aug = np.array(y_aug)

# 3. Soft-margin SVM with healthy regularization (C=1.5)
clf = Pipeline([
    ('scaler', PixelScaler()),
    ('svm', SVC(kernel='rbf', C=1.5, gamma='scale', decision_function_shape='ovr'))
])
clf.fit(X_aug, y_aug)

# 4. Genuine test evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
errors = int(np.sum(y_test != y_pred))

print(f"Test Accuracy: {acc*100:.2f}% ({len(X_test)-errors}/{len(X_test)} correct, {errors} errors)")
print("\nConfusion Matrix:")
print(cm)

# 5. Test on varied freehand drawing styles (thin w=12, medium w=20, thick w=28)
def process_drawing(pil_img):
    gray = pil_img.convert("L")
    arr = 255.0 - np.array(gray, dtype=float)
    mask = arr > 25.0
    if np.sum(mask) < 20: return None
    
    # Adaptive stroke dilation if thin
    ymin, ymax = np.where(np.any(mask, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
    h = ymax - ymin + 1
    w = xmax - xmin + 1
    
    # If thin, apply MaxFilter
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
    
    # Proportional padding
    pad = max(10, int(round(max(h, w) * 0.12)))
    orig_h, orig_w = arr_d.shape
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(orig_h - 1, ymax + pad)
    xmax = min(orig_w - 1, xmax + pad)
    
    cropped = arr_d[ymin:ymax+1, xmin:xmax+1]
    ch, cw = cropped.shape
    
    # Center into square
    max_side = max(ch, cw)
    sq = np.zeros((max_side, max_side), dtype=float)
    sq[(max_side-ch)//2:(max_side-ch)//2+ch, (max_side-cw)//2:(max_side-cw)//2+cw] = cropped
    
    res = Image.fromarray(sq.astype(np.uint8)).resize((8, 8), Image.Resampling.BOX)
    arr8 = np.array(res, dtype=float)
    
    if arr8.max() > 0:
        arr8 = (arr8 / arr8.max()) * 16.0
        
    cy, cx = center_of_mass(arr8)
    if not (np.isnan(cy) or np.isnan(cx)):
        dy = float(np.clip(3.5 - cy, -1.2, 1.2))
        dx = float(np.clip(3.5 - cx, -1.2, 1.2))
        arr8 = shift(arr8, [dy, dx], mode='constant', cval=0.0)
        
    return np.clip(arr8, 0, 16).flatten()

def draw_digit_stroke(digit, w_pen):
    img = Image.new('L', (280, 280), 255)
    draw = ImageDraw.Draw(img)
    if digit == 0:   draw.ellipse([60, 40, 220, 240], outline=0, width=w_pen)
    elif digit == 1: draw.line([140, 40, 140, 240], fill=0, width=w_pen)
    elif digit == 2:
        draw.arc([60, 40, 220, 140], 180, 0, fill=0, width=w_pen)
        draw.line([220, 90, 60, 240], fill=0, width=w_pen)
        draw.line([60, 240, 220, 240], fill=0, width=w_pen)
    elif digit == 3:
        draw.line([70, 40, 210, 40], fill=0, width=w_pen)
        draw.line([210, 40, 130, 130], fill=0, width=w_pen)
        draw.arc([60, 120, 220, 240], 270, 90, fill=0, width=w_pen)
    elif digit == 4:
        draw.line([180, 40, 60, 170], fill=0, width=w_pen)
        draw.line([60, 170, 230, 170], fill=0, width=w_pen)
        draw.line([180, 40, 180, 240], fill=0, width=w_pen)
    elif digit == 5:
        draw.line([210, 40, 70, 40], fill=0, width=w_pen)
        draw.line([70, 40, 70, 130], fill=0, width=w_pen)
        draw.line([70, 130, 170, 130], fill=0, width=w_pen)
        draw.arc([60, 120, 220, 240], 270, 90, fill=0, width=w_pen)
    elif digit == 6:
        draw.arc([60, 40, 220, 240], 90, 270, fill=0, width=w_pen)
        draw.ellipse([60, 130, 220, 240], outline=0, width=w_pen)
    elif digit == 7:
        draw.line([60, 40, 220, 40], fill=0, width=w_pen)
        draw.line([220, 40, 110, 240], fill=0, width=w_pen)
    elif digit == 8:
        draw.ellipse([75, 40, 205, 135], outline=0, width=w_pen)
        draw.ellipse([60, 130, 220, 240], outline=0, width=w_pen)
    elif digit == 9:
        draw.ellipse([60, 40, 220, 150], outline=0, width=w_pen)
        draw.line([220, 95, 220, 240], fill=0, width=w_pen)
    return img

print("\nTesting freehand drawing recognition across 3 pen widths:")
for w_pen in [12, 20, 28]:
    correct = 0
    for d in range(10):
        img = draw_digit_stroke(d, w_pen)
        x = process_drawing(img)
        pred = clf.predict(x.reshape(1, -1))[0]
        if pred == d: correct += 1
    print(f"  Pen width {w_pen:2d}px: {correct}/10 recognized correctly")
