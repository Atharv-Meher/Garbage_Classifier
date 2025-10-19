import os
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import json
import time

# ---------------------------
# Configuration
# ---------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_TO_CATEGORY = {
    "bottle": "recyclable",
    "cup": "recyclable",
    "wine glass": "recyclable",
    "banana": "biodegradable",
    "apple": "biodegradable",
    "orange": "biodegradable",
    "chair": "non_recyclable",
    "sofa": "non_recyclable",
    "book": "recyclable",
    "newspaper": "recyclable",
    "paper": "recyclable",
    "glass": "recyclable",
    "tin": "recyclable"
}
CONF_THRESHOLD = 0.25

# ---------------------------
# Utilities
# ---------------------------
def pil_to_cv2(image: Image.Image):
    img = np.array(image.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def draw_boxes(img_bgr, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = det['label']
        conf = det['confidence']
        color = (0, 255, 0)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, f"{label} {conf:.2f}", (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_bgr

def map_detection_to_category(label):
    return CLASS_TO_CATEGORY.get(label.lower(), "unknown")

def aggregate_category(detections):
    scores = {}
    for d in detections:
        cat = d['category']
        scores[cat] = scores.get(cat, 0.0) + d['confidence']
    if not scores:
        return {"predicted_label": "unknown", "confidence": 0.0}
    total = sum(scores.values())
    norm_scores = {k: v/total for k,v in scores.items()}
    best = max(norm_scores.items(), key=lambda kv: kv[1])
    return {"predicted_label": best[0], "confidence": round(best[1],4), "all_scores": {k: round(v,4) for k,v in norm_scores.items()}}

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # nano pretrained

@app.route("/", methods=["GET", "POST"])
def index():
    result_json = None
    output_image_url = None

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        image = Image.open(file).convert("RGB")
        img_bgr = pil_to_cv2(image)

        # Run YOLO
        start = time.time()
        results = model.predict(source=np.array(image), verbose=False)
        elapsed = time.time() - start
        res = results[0]

        class_names = res.names
        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes.data.tolist():  # xyxy, conf, cls
                x1, y1, x2, y2, conf, cls = box
                conf = float(conf)
                cls = int(cls)
                label = class_names.get(cls, str(cls))
                if conf < CONF_THRESHOLD:
                    continue
                category = map_detection_to_category(label)
                det = {
                    "label": label,
                    "confidence": round(conf,4),
                    "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                    "category": category
                }
                detections.append(det)

        # draw
        img_with_boxes = draw_boxes(img_bgr.copy(), detections)
        pil_out = cv2_to_pil(img_with_boxes)
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        pil_out.save(output_path)
        output_image_url = output_path

        agg = aggregate_category(detections)
        result_json = {
            "detections": detections,
            "aggregation": agg,
            "inference_time_sec": round(elapsed,2)
        }

    return render_template("index.html", output_image_url=output_image_url, result_json=result_json)

if __name__ == "__main__":
    app.run(debug=True)
