
import gradio as gr
from ultralytics import YOLO
import os
import tempfile
import subprocess
import cv2
import pandas as pd
from collections import defaultdict
from PIL import Image
import numpy as np

model = YOLO("yolov8n.pt")

def download_video_yt_dlp(url, output_path):
    try:
        command = ["yt-dlp", "-f", "mp4", "-o", output_path, url]
        subprocess.run(command, check=True)
        return output_path, None
    except subprocess.CalledProcessError as e:
        return None, f"[yt-dlp Download Error] {str(e)}"

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "[Error] Could not open the video file."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    object_counts = defaultdict(int)
    object_snapshots = {}
    snapshot_frames = {}

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            object_counts[label] += 1

            if label not in object_snapshots:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                object_image = frame[y1:y2, x1:x2]
                object_snapshots[label] = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)
                snapshot_frames[label] = frame_idx

    cap.release()
    summary_data = [{"Object": label, "Count": count, "Frame": snapshot_frames.get(label, -1)} for label, count in object_counts.items()]
    df = pd.DataFrame(summary_data).sort_values(by="Count", ascending=False).reset_index(drop=True)
    return df, object_snapshots, None

def app_main(video_url):
    if not video_url.startswith("https://www.youtube.com/watch?v="):
        return None, None, "[Input Error] Please enter a valid YouTube video link."

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        video_path, error = download_video_yt_dlp(video_url, video_path)
        if error:
            return None, None, error

        df, snapshots, error = detect_objects(video_path)
        if error:
            return None, None, error

        snapshot_images = [Image.fromarray(snapshots[obj]) for obj in df["Object"] if obj in snapshots]
        return snapshot_images, df, None

iface = gr.Interface(
    fn=app_main,
    inputs=gr.Textbox(label="YouTube Video URL", placeholder="Paste a YouTube video link..."),
    outputs=[
        gr.Gallery(label="Detected Object Snapshots", show_label=True, columns=4),
        gr.Dataframe(label="Object Detection Summary"),
        gr.Textbox(label="Error Log")
    ],
    title="🎥 YouTube Object Detection App (YOLOv8n + yt-dlp)",
    description="Detects objects from YouTube videos using YOLOv8n. Paste a YouTube link to begin.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
