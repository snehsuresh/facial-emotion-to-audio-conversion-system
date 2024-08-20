import cv2
import time
from collections import defaultdict
from expressiondetection.yolo import yolo_detect
from expressiondetection.config.model_config import DETECTION_INTERVAL
from expressiondetection.utils.main_utils import speak_text
import queue


def predict_live(frame_queue):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    last_detection_time = time.time()
    last_audio_time = time.time() - 10  # Initial value to ensure first audio is played
    detection_counts = defaultdict(int)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                continue

            faces = face_cascade.detectMultiScale(
                frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            current_time = time.time()

            if current_time - last_detection_time > DETECTION_INTERVAL:
                # Reset detection counts for new interval
                detection_counts.clear()

                for x, y, w, h in faces:
                    face_region = frame[y : y + h, x : x + w]
                    detected_label = yolo_detect(face_region)

                    if detected_label:
                        detection_counts[detected_label] += 1

                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        label = f"{detected_label}"
                        cv2.putText(
                            frame,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )

                # Find the most frequent label
                if detection_counts:
                    most_frequent_label = max(
                        detection_counts, key=detection_counts.get
                    )

                    # Speak the most frequent label if 10 seconds have passed
                    if current_time - last_audio_time > 10:
                        speak_text(f"The person is {most_frequent_label}")
                        last_audio_time = current_time

                last_detection_time = current_time

            if not frame_queue.full():
                frame_queue.put(frame)

    finally:
        cap.release()
        print("Camera stopped!")
