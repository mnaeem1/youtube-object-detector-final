
---
title: "YouTube Object Detector"
emoji: "🎥"
colorFrom: "blue"
colorTo: "green"
sdk: gradio
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---

# 🎥 YouTube Object Detection App (YOLOv8n + Hugging Face/Streamlit)

This app downloads a public YouTube video using `yt-dlp`, runs object detection using YOLOv8n (lightweight), and shows:
- 📸 Snapshots per detected object
- 📊 Table of object counts and frame numbers

## 🚀 How to Use (Streamlit Cloud or Hugging Face)

- Ensure `yolov8n.pt` is uploaded into the repo root (already included)
- App runs without needing to download model weights from internet
