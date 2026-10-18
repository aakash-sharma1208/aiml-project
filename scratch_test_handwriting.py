import os
import model_utils
from PIL import Image, ImageDraw, ImageFont
import numpy as np

fonts = ["arial.ttf", "calibri.ttf", "comic.ttf", "times.ttf"]

for font_name in fonts:
    print(f"\n--- Testing with font: {font_name} ---")
    try:
        font = ImageFont.truetype(font_name, 170)
    except Exception as e:
        print(f"Font {font_name} not found: {e}")
        continue

    for d in range(10):
        img = Image.new("L", (280, 280), color=255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), str(d), font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (280 - w) // 2 - bbox[0]
        y = (280 - h) // 2 - bbox[1]
        draw.text((x, y), str(d), fill=0, font=font)

        feat = model_utils.process_pil_image(img)
        if feat is not None:
            res = model_utils.predict_from_features(feat.reshape(1, -1))
            pred = res["predicted_digit"]
            match = "[OK]" if pred == d else f"[FAIL] (got {pred})"
            print(f"Digit {d}: {match}")
        else:
            print(f"Digit {d}: BLANK")
