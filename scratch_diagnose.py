import model_utils
from test_simulated_drawings import draw_digit_stroke

for d in [3, 4, 7]:
    img = draw_digit_stroke(d)
    feat = model_utils.process_pil_image(img).reshape(8, 8)
    print(f"\n=== Drawn Digit {d} (preprocessed 8x8) ===")
    for row in feat:
        print("".join(f"{int(val):2d} " if val > 0 else " . " for val in row))
    res = model_utils.predict_from_features(feat.reshape(1, -1))
    print(f"Predicted: {res['predicted_digit']}")
    pipeline = model_utils.get_pipeline()
    scores = pipeline.decision_function(feat.reshape(1, -1))[0]
    for c, s in enumerate(scores):
        print(f"  Class {c}: {s:.2f}")
