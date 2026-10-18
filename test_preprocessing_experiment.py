import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import model_utils

def preprocess_test(pil_img: Image.Image, target_h=7.5, pad_ratio=0.04):
    gray = pil_img.convert("L")
    arr = 255.0 - np.array(gray, dtype=float)

    mask = arr > 25.0
    if np.sum(mask) < 20:
        return None

    # Adaptive stroke dilation based on stroke thickness
    density = np.sum(mask) / (np.max(np.where(np.any(mask, axis=1))[0]) - np.min(np.where(np.any(mask, axis=1))[0]) + 1)
    img_inv = Image.fromarray(arr.astype(np.uint8))
    if density < 15:
        img_inv = img_inv.filter(ImageFilter.MaxFilter(7))
    elif density < 25:
        img_inv = img_inv.filter(ImageFilter.MaxFilter(5))
    elif density < 35:
        img_inv = img_inv.filter(ImageFilter.MaxFilter(3))

    arr_d = np.array(img_inv, dtype=float)
    mask_d = arr_d > 25.0
    ys, xs = np.where(mask_d)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    h = ymax - ymin + 1
    w = xmax - xmin + 1

    # Crop tight to the dilated digit
    cropped = arr_d[ymin:ymax+1, xmin:xmax+1]

    # In digits dataset: height is ~7-8, width is ~5-6
    # Let's map height to 7 or 8
    # Keep aspect ratio, but ensure width is at least 3
    aspect = w / h
    # Target in 8x8:
    new_h = 7.0
    new_w = min(6.5, max(3.0, new_h * aspect))

    # Resize cropped using BOX filter to high-res, then place onto 8x8 canvas
    # Let's do it cleanly by placing cropped onto a canvas with margins matching dataset:
    # Top margin ~ 0.5, bottom margin ~ 0.5 (so height occupies 7 rows)
    # Left/right centered
    margin_top = int(round(h * 0.05))
    margin_side = int(round(w * 0.08))
    
    # We want final 8x8: so let canvas height = h + 2*margin_top
    # canvas width = max(w + 2*margin_side, int((h + 2*margin_top) * 0.85))
    total_h = h + 2 * margin_top
    total_w = max(w + 2 * margin_side, int(round(total_h * 0.85)))
    
    canvas = np.zeros((total_h, total_w), dtype=float)
    cy = (total_h - h) // 2
    cx = (total_w - w) // 2
    canvas[cy:cy+h, cx:cx+w] = cropped

    c_img = Image.fromarray(canvas.astype(np.uint8))
    res8 = c_img.resize((8, 8), Image.Resampling.BOX)
    arr8 = np.array(res8, dtype=float)

    # Dynamic contrast stretch
    if arr8.max() > 0:
        arr8 = (arr8 / arr8.max()) * 16.0

    return np.clip(arr8, 0.0, 16.0).flatten().astype(np.float32)

fonts = ["arial.ttf", "calibri.ttf", "comic.ttf", "times.ttf"]
pipeline = model_utils.get_pipeline()

for font_name in fonts:
    print(f"\n--- Testing with font: {font_name} ---")
    font = ImageFont.truetype(font_name, 170)
    for d in range(10):
        img = Image.new("L", (280, 280), color=255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), str(d), font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (280 - w) // 2 - bbox[0]
        y = (280 - h) // 2 - bbox[1]
        draw.text((x, y), str(d), fill=0, font=font)

        feat = preprocess_test(img)
        pred = pipeline.predict(feat.reshape(1, -1))[0]
        match = "[OK]" if pred == d else f"[FAIL] (got {pred})"
        print(f"Digit {d}: {match}")
