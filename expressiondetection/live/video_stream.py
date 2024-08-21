import cv2
from collections import defaultdict
from expressiondetection.yolo import yolo_detect
from expressiondetection.utils.main_utils import speak_text


def process_frame(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame, faces


def predict_emotion_from_frame(frame, face):
    detection_counts = defaultdict(int)

    for x, y, w, h in face:
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
            speak_text(f"The person is {detected_label}")
            return detected_label
        else:
            speak_text("Something's wrong")

    return "No Expression"
