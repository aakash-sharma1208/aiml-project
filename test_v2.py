import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import center_of_mass, shift
import joblib

m = joblib.load('models/svm_pipeline.joblib')
pipe = m['pipeline']

def process_pil_image_v2(pil_img: Image.Image) -> np.ndarray:
    gray = pil_img.convert("L")
    arr = 255.0 - np.array(gray, dtype=float)
    
    # 1. Clean thresholding
    mask = arr > 25.0
    if np.sum(mask) < 20:
        return None
        
    # 2. Check stroke thickness: if thin, dilate with MaxFilter
    # Average thickness = number of active pixels / perimeter or height
    ymin, ymax = np.where(np.any(mask, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
    h = ymax - ymin + 1
    w = xmax - xmin + 1
    if h < 12 and w < 12:
        return None

    # Apply morphological dilation if strokes are thin
    ink_count = np.sum(mask)
    density = ink_count / (h + w)
    # If density is low (thin stroke), dilate by 3x3 or 5x5
    img_inv = Image.fromarray(arr.astype(np.uint8))
    if density < 12:
        img_inv = img_inv.filter(ImageFilter.MaxFilter(5))
    elif density < 18:
        img_inv = img_inv.filter(ImageFilter.MaxFilter(3))
        
    arr_dil = np.array(img_inv, dtype=float)
    mask_dil = arr_dil > 25.0
    ymin, ymax = np.where(np.any(mask_dil, axis=1))[0][[0, -1]]
    xmin, xmax = np.where(np.any(mask_dil, axis=0))[0][[0, -1]]
    h = ymax - ymin + 1
    w = xmax - xmin + 1

    cropped = arr_dil[ymin:ymax+1, xmin:xmax+1]
    
    # Smart aspect ratio scaling:
    # Target inner box is roughly 6.2 x 6.2
    c_img = Image.fromarray(cropped.astype(np.uint8))
    aspect = h / max(1, w)
    
    if aspect > 2.2:
        # Tall digit like 1: height=6, width=2.5 to 3
        target_h = 6
        target_w = 3
    elif aspect < 0.45:
        # Very wide stroke: width=6, height=3
        target_w = 6
        target_h = 3
    elif w > h:
        target_w = 6
        target_h = max(3, int(round(6 * h / w)))
    else:
        target_h = 6
        target_w = max(3, int(round(6 * w / h)))

    res = c_img.resize((target_w, target_h), Image.Resampling.BOX)
    res_arr = np.array(res, dtype=float)
    
    # Place in 8x8 canvas centered
    canvas8 = np.zeros((8, 8), dtype=float)
    sy = (8 - target_h) // 2
    sx = (8 - target_w) // 2
    canvas8[sy:sy+target_h, sx:sx+target_w] = res_arr
    
    # Contrast normalization
    if canvas8.max() > 0:
        canvas8 = (canvas8 / canvas8.max()) * 16.0
        
    # Center of mass alignment
    cy, cx = center_of_mass(canvas8)
    if not (np.isnan(cy) or np.isnan(cx)):
        dy = float(np.clip(3.5 - cy, -1.2, 1.2))
        dx = float(np.clip(3.5 - cx, -1.2, 1.2))
        canvas8 = shift(canvas8, [dy, dx], mode='constant', cval=0.0)
        
    return np.clip(canvas8, 0.0, 16.0).flatten()

print("Testing v2 preprocessing on varied writing styles...")
# draw_canvas is defined inline

for w_val in [10, 18, 28]:
    print(f"\n--- Testing stroke width = {w_val}px ---")
    correct = 0
    for d in range(10):
        img = Image.new('L', (280, 280), 255)
        draw = ImageDraw.Draw(img)
        # Draw digits with pen thickness = w_val
        if d == 0:   draw.ellipse([60, 40, 220, 240], outline=0, width=w_val)
        elif d == 1: draw.line([140, 40, 140, 240], fill=0, width=w_val)
        elif d == 2:
            draw.arc([60, 40, 220, 140], 180, 0, fill=0, width=w_val)
            draw.line([220, 90, 60, 240], fill=0, width=w_val)
            draw.line([60, 240, 220, 240], fill=0, width=w_val)
        elif d == 3:
            draw.line([70, 40, 210, 40], fill=0, width=w_val)
            draw.line([210, 40, 130, 130], fill=0, width=w_val)
            draw.arc([60, 120, 220, 240], 270, 90, fill=0, width=w_val)
        elif d == 4:
            draw.line([180, 40, 60, 170], fill=0, width=w_val)
            draw.line([60, 170, 230, 170], fill=0, width=w_val)
            draw.line([180, 40, 180, 240], fill=0, width=w_val)
        elif d == 5:
            draw.line([210, 40, 70, 40], fill=0, width=w_val)
            draw.line([70, 40, 70, 130], fill=0, width=w_val)
            draw.line([70, 130, 170, 130], fill=0, width=w_val)
            draw.arc([60, 120, 220, 240], 270, 90, fill=0, width=w_val)
        elif d == 6:
            draw.arc([60, 40, 220, 240], 90, 270, fill=0, width=w_val)
            draw.ellipse([60, 130, 220, 240], outline=0, width=w_val)
        elif d == 7:
            draw.line([60, 40, 220, 40], fill=0, width=w_val)
            draw.line([220, 40, 110, 240], fill=0, width=w_val)
        elif d == 8:
            draw.ellipse([75, 40, 205, 135], outline=0, width=w_val)
            draw.ellipse([60, 130, 220, 240], outline=0, width=w_val)
        elif d == 9:
            draw.ellipse([60, 40, 220, 150], outline=0, width=w_val)
            draw.line([220, 95, 220, 240], fill=0, width=w_val)
            
        x = process_pil_image_v2(img)
        pred = pipe.predict(x.reshape(1, -1))[0]
        match = (pred == d)
        if match: correct += 1
        print(f"Digit {d} (width={w_val}): Pred={pred} {'[OK]' if match else '[FAIL]'}")
    print(f"Score for width={w_val}: {correct}/10")
