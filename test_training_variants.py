import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.ndimage import shift, rotate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from model_utils import PixelScaler

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

aug_X = [X_train]
aug_y = [y_train]

# Translations
for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    sh = np.zeros_like(X_train)
    for i, s in enumerate(X_train):
        sh[i] = shift(s.reshape(8, 8), [dy, dx], mode='constant', cval=0.0).flatten()
    aug_X.append(sh)
    aug_y.append(y_train)

# Small rotations (-8, +8 degrees)
for angle in [-8, 8]:
    rot = np.zeros_like(X_train)
    for i, s in enumerate(X_train):
        rot[i] = rotate(s.reshape(8, 8), angle, reshape=False, mode='constant', cval=0.0).flatten()
    aug_X.append(rot)
    aug_y.append(y_train)

X_train_aug = np.vstack(aug_X)
y_train_aug = np.concatenate(aug_y)

pipe = Pipeline([
    ('scaler', PixelScaler()),
    ('svm', SVC(kernel='rbf', C=3.0, gamma='scale', decision_function_shape='ovr', random_state=42))
])

pipe.fit(X_train_aug, y_train_aug)
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
errors = int(np.sum(y_pred != y_test))
print(f"Augmented samples: {len(X_train_aug)}")
print(f"Test Accuracy: {acc*100:.2f}% ({len(y_test) - errors}/{len(y_test)} correct, {errors} errors)")
