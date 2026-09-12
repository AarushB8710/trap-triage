# 🦉 TrapTriage

**Automatic screening for camera-trap wildlife photos.** Upload a batch of images, and TrapTriage filters out blanks, flags real detections, and gives conservation researchers a live triage dashboard instead of hours of manual review.

Built for AnimalHack 2026.

## The problem

Camera traps are one of the most widely used tools in conservation research, but manual review of the images they produce is a well-documented bottleneck. Studies show that up to **70% of camera-trap images are blank false triggers** (wind, rain, shifting shadows), and researchers routinely lose weeks sorting through thousands of photos before any real analysis can begin.

## What it does

1. **Upload** a `.zip` of camera-trap photos through a simple web interface
2. **Detect** — every image is run through Microsoft's open-source [MegaDetectorV6](https://github.com/microsoft/CameraTraps) model (via [PyTorch-Wildlife](https://github.com/microsoft/CameraTraps))
3. **Triage** — a confidence-threshold rule (written for this project) separates:
   - ✅ Confirmed detections
   - ⚠️ Low-confidence detections flagged for human review
   - 🌫️ Blank images, filtered out automatically
4. **Visualize** — bounding boxes are drawn on each image, and a live dashboard shows totals, blank-filter rate, and per-image detail

## Architecture

```
Upload (.zip) → Flask backend → MegaDetectorV6 inference (per image)
                                        ↓
                     Confidence-based triage logic (this project)
                                        ↓
                  Bounding-box rendering + results dashboard (this project)
```

MegaDetectorV6 provides the raw animal/blank detection — the standard, widely-used building block for this kind of work (it's used in 80+ real conservation programs). Everything from the confidence-threshold review logic, to the tallying, to the dashboard itself, was built specifically for this project.

## Setup

### 1. Clone this repo
```bash
git clone https://github.com/<your-username>/trap-triage.git
cd trap-triage
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the model weights
The detection model weights (~50MB) aren't included in this repo (too large for git). Download them once:

```bash
curl -L -o MDV6-yolov9-c.pt "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1"
```

Place `MDV6-yolov9-c.pt` in the same folder as `app.py`.

### 4. Run it
```bash
python3 app.py
```

Open **http://127.0.0.1:5000** in your browser, upload a `.zip` of camera-trap images, and click "Run Detection."

## Tech stack

Python · Flask · PyTorch · MegaDetectorV6 · PyTorch-Wildlife · Pillow · HTML/CSS/JS

## Future work

- Species-level classification on top of animal/blank detection
- Trend tracking across batches over time (sightings per week/month)
- Hosted version so researchers don't need local setup

## Credits

Detection powered by [MegaDetectorV6](https://github.com/microsoft/CameraTraps), developed by Microsoft AI for Good.
