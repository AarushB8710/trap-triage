"""
Camera-Trap Triage Platform
----------------------------
A local web app: upload a batch of camera-trap photos as a .zip, the app
runs MegaDetectorV6 on every image, applies a confidence-based triage rule,
and renders a live dashboard -- blanks filtered out, confirmed detections
tallied, uncertain ones flagged for human review.

RUN LOCALLY:
    1. pip install flask PytorchWildlife pillow
    2. Put MDV6-yolov9-c.pt in the same folder as this file
       (download once from https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1 )
    3. python app.py
    4. Open http://127.0.0.1:5000 in your browser
"""

import os
import json
import shutil
import zipfile
import uuid
from flask import Flask, request, render_template, send_from_directory, redirect, url_for

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
WEIGHTS_PATH = os.path.join(BASE_DIR, "MDV6-yolov9-c.pt")
REVIEW_THRESHOLD = 0.5
CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

_model = None
def get_model():
    """Load MegaDetectorV6 once and reuse it across requests."""
    global _model
    if _model is None:
        from PytorchWildlife.models import detection as pw_detection
        _model = pw_detection.MegaDetectorV6(weights=WEIGHTS_PATH, version="MDV6-yolov9-c")
    return _model


def run_pipeline(batch_dir):
    """Run detection + triage on every image in batch_dir/images, write
    annotated images into batch_dir/annotated, return the summary dict."""
    from PIL import Image, ImageDraw

    model = get_model()
    images_dir = os.path.join(batch_dir, "images")
    annotated_dir = os.path.join(batch_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    summary = {
        "total_images": len(image_files),
        "blank_images": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "needs_review": 0,
        "by_class": {},
    }
    per_image_rows = []

    for fname in image_files:
        path = os.path.join(images_dir, fname)
        try:
            r = model.single_image_detection(path)
            det = r["detections"]
            confs = det.confidence.tolist() if hasattr(det, "confidence") else []
            boxes = det.xyxy.tolist() if hasattr(det, "xyxy") else []
            class_ids = det.class_id.tolist() if hasattr(det, "class_id") else []
        except Exception as e:
            confs, boxes, class_ids = [], [], []
            print(f"Error on {fname}: {e}")

        if not confs:
            summary["blank_images"] += 1
            per_image_rows.append({
                "img_id": fname, "status": "blank", "best_conf": None,
                "n_detections": 0, "classes": []
            })
            shutil.copy(path, os.path.join(annotated_dir, fname))
            continue

        summary["images_with_detections"] += 1
        summary["total_detections"] += len(confs)
        best_conf = max(confs)
        status = "confident" if best_conf >= REVIEW_THRESHOLD else "needs_review"
        if status == "needs_review":
            summary["needs_review"] += 1

        classes_present = sorted(set(CLASS_NAMES.get(c, f"class_{c}") for c in class_ids))
        for cid in class_ids:
            cname = CLASS_NAMES.get(cid, f"class_{cid}")
            summary["by_class"][cname] = summary["by_class"].get(cname, 0) + 1

        per_image_rows.append({
            "img_id": fname, "status": status, "best_conf": round(best_conf, 3),
            "n_detections": len(confs), "classes": classes_present
        })

        # draw boxes onto a copy of the image
        im = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for box, conf, cid in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = box
            color = (255, 60, 60) if conf < REVIEW_THRESHOLD else (60, 220, 100)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
            label = f"{CLASS_NAMES.get(cid, cid)} {conf:.2f}"
            draw.rectangle([x1, max(0, y1 - 32), x1 + 10 + len(label) * 13, y1], fill=color)
            draw.text((x1 + 5, y1 - 30), label, fill=(0, 0, 0))
        im.save(os.path.join(annotated_dir, fname))

    result = {"summary": summary, "per_image": per_image_rows}
    with open(os.path.join(batch_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("zipfile")
    if not file or not file.filename.lower().endswith(".zip"):
        return redirect(url_for("index"))

    batch_id = uuid.uuid4().hex[:10]
    batch_dir = os.path.join(UPLOAD_DIR, batch_id)
    images_dir = os.path.join(batch_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    zip_path = os.path.join(batch_dir, "upload.zip")
    file.save(zip_path)

    # Extract, ignoring junk files (e.g. __MACOSX, dotfiles)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            base = os.path.basename(member)
            if not base or member.startswith("__MACOSX") or base.startswith("."):
                continue
            if base.lower().endswith((".jpg", ".jpeg", ".png")):
                with z.open(member) as src, open(os.path.join(images_dir, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)

    result = run_pipeline(batch_dir)
    return redirect(url_for("results", batch_id=batch_id))


@app.route("/results/<batch_id>")
def results(batch_id):
    batch_dir = os.path.join(UPLOAD_DIR, batch_id)
    summary_path = os.path.join(batch_dir, "summary.json")
    if not os.path.exists(summary_path):
        return redirect(url_for("index"))
    with open(summary_path) as f:
        data = json.load(f)
    return render_template("results.html", batch_id=batch_id,
                            summary=data["summary"], rows=data["per_image"])


@app.route("/uploads/<batch_id>/annotated/<filename>")
def annotated_image(batch_id, filename):
    return send_from_directory(os.path.join(UPLOAD_DIR, batch_id, "annotated"), filename)


if __name__ == "__main__":
    print("Loading model (first run downloads nothing -- uses local weights)...")
    get_model()
    print("Model ready. Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=False, port=5000)
