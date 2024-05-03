import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def save_uploaded_file(uploaded_file):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "uploaded_video.mp4"

def perform_object_detection(video_file, output_file):
    model = YOLO('best.pt').load('best.pt')
    model.cpu()

    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    detected_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            names = result.names
            classes = result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                xmin, ymin, xmax, ymax = box.astype(int)
                label = names[cls]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        detected_frames.append(frame)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(frame_pil, channels="RGB", use_column_width=True)

    cap.release()
    out.release()

    # Create a video of detected frames
    if len(detected_frames) > 0:
        detected_output_file = "detected_frames.mp4"
        out = cv2.VideoWriter(detected_output_file, fourcc, fps, (width, height))
        for frame in detected_frames:
            out.write(frame)
        out.release()
        st.success(f"Detected frames saved as {detected_output_file}")
def main():
    st.title("Violence Detection on Video with YOLO")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        video_file_path = save_uploaded_file(uploaded_file)
        st.success(f"Uploaded video saved as {video_file_path}")

        output_file = "output.mp4"
        perform_object_detection(video_file_path, output_file)
        st.success(f"Processed video saved as {output_file}")
        with open('detected_frames.mp4', 'rb') as f:
          st.download_button('Download Zip', f, file_name='video.mp4')
        
            

if __name__ == "__main__":
    main()
