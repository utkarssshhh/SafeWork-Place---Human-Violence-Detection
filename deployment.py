import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, ImageSequenceClip

def save_uploaded_file(uploaded_file):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "uploaded_video.mp4"

def perform_object_detection(video_file, output_file):
    model = YOLO('best.pt').load('best.pt')
    model.cpu()

    clip = VideoFileClip(video_file)
    fps = clip.fps
    width, height = clip.size

    detected_frames = []

    for frame in clip.iter_frames():
        img = Image.fromarray(frame)
        results = model.predict(np.array(img))
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            names = result.names
            classes = result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                xmin, ymin, xmax, ymax = box.astype(int)
                label = names[cls]
                img = draw_bbox(img, xmin, ymin, xmax, ymax, label)

        detected_frames.append(np.array(img))

    # Write the detected frames to a video file
    if len(detected_frames) > 0:
        detected_output_file = "detected_frames.mp4"
        detected_clip = ImageSequenceClip(detected_frames, fps=fps)
        detected_clip.write_videofile(detected_output_file, fps=fps)
        st.success(f"Detected frames saved as {detected_output_file}")

def draw_bbox(img, xmin, ymin, xmax, ymax, label):
    # Draw bounding box and label on the image
    bbox_color = (0, 255, 0)
    font_color = (0, 255, 0)
    thickness = 2
    font_scale = 0.5
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    draw.rectangle([xmin, ymin, xmax, ymax], outline=bbox_color, width=thickness)
    draw.text((xmin, ymin - 10), label, fill=font_color)
    return img

def main():
    st.title("Object Detection on Video with YOLO")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        video_file_path = save_uploaded_file(uploaded_file)
        st.success(f"Uploaded video saved as {video_file_path}")

        output_file = "output.mp4"
        perform_object_detection(video_file_path, output_file)
        st.success(f"Processed video saved as {output_file}")
        with open('detected_frames.mp4', 'rb') as v:
            st.video(v)

if __name__ == "__main__":
    main()
