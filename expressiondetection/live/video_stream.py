import cv2
from collections import defaultdict
from expressiondetection.yolo import yolo_detect


def process_frame(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    # print("Faces detected:", faces)

    # if len(faces) == 0:
    #     # print("No faces detected")

    for x, y, w, h in faces:
        # Draw rectangle around the face
        # print(f"Drawing rectangle at ({x}, {y}), width {w}, height {h}")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the image for debugging (optional, only works if you have a GUI)

    return frame, faces


def predict_emotion_from_frame(frame, face):
    detection_counts = defaultdict(int)
    print("Predicting emotion==")
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
            # a = speak_text(f"The person is {detected_label}")
            return detected_label
        else:
            return "I don't know!"

    return "No Expression"
