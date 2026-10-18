import numpy as np
from PIL import Image, ImageDraw
import model_utils

def draw_digit_stroke(digit):
    img = Image.new("L", (280, 280), color=255)
    draw = ImageDraw.Draw(img)
    w = 22

    if digit == 0:
        # Oval
        draw.ellipse([70, 40, 210, 240], outline=0, width=w)
    elif digit == 1:
        # Vertical stroke slightly slanted
        draw.line([(145, 45), (135, 235)], fill=0, width=w)
    elif digit == 2:
        # Top arc, diagonal down to left, horizontal base
        draw.arc([75, 40, 205, 140], start=180, end=0, fill=0, width=w)
        draw.line([(205, 90), (75, 235)], fill=0, width=w)
        draw.line([(75, 235), (205, 235)], fill=0, width=w)
    elif digit == 3:
        # Top arc and bottom arc
        draw.arc([80, 40, 200, 140], start=200, end=40, fill=0, width=w)
        draw.arc([80, 135, 200, 240], start=300, end=140, fill=0, width=w)
    elif digit == 4:
        # Vertical-left, horizontal bar, vertical stem
        draw.line([(85, 50), (75, 160)], fill=0, width=w)
        draw.line([(75, 160), (210, 160)], fill=0, width=w)
        draw.line([(175, 50), (175, 235)], fill=0, width=w)
    elif digit == 5:
        # Top bar, vertical drop, round bottom
        draw.line([(195, 45), (85, 45)], fill=0, width=w)
        draw.line([(85, 45), (85, 130)], fill=0, width=w)
        draw.arc([80, 115, 205, 240], start=280, end=140, fill=0, width=w)
    elif digit == 6:
        # Loop with curve
        draw.arc([75, 40, 195, 235], start=90, end=270, fill=0, width=w)
        draw.ellipse([75, 125, 200, 235], outline=0, width=w)
    elif digit == 7:
        # Top bar and diagonal stem
        draw.line([(75, 45), (205, 45)], fill=0, width=w)
        draw.line([(205, 45), (115, 235)], fill=0, width=w)
    elif digit == 8:
        # Two stacked loops
        draw.ellipse([85, 40, 195, 140], outline=0, width=w)
        draw.ellipse([80, 130, 200, 240], outline=0, width=w)
    elif digit == 9:
        # Top loop and right stem
        draw.ellipse([80, 40, 200, 150], outline=0, width=w)
        draw.line([(200, 95), (200, 235)], fill=0, width=w)

    return img

print("--- Testing Simulated Hand-Drawn Strokes with Current Preprocessing ---")
for d in range(10):
    img = draw_digit_stroke(d)
    feat = model_utils.process_pil_image(img)
    if feat is not None:
        res = model_utils.predict_from_features(feat.reshape(1, -1))
        pred = res["predicted_digit"]
        status = "[OK]" if pred == d else f"[FAIL] (got {pred})"
        print(f"Drawn Digit {d}: {status}")
    else:
        print(f"Drawn Digit {d}: BLANK")
