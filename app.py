from flask import Flask, render_template, Response, jsonify
from expressiondetection.live.video_stream import (
    process_frame,
    predict_emotion_from_frame,
)
import threading
import collections
import cv2

app = Flask(__name__)

# Variables for camera
camera_thread = None
camera_running = False
frame_deque = collections.deque(maxlen=1)  # Only keep the most recent frame
face_deque = collections.deque(maxlen=1)  # Only keep the most recent face


def start_camera():
    global camera_running
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    camera_running = True
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue

        # Process frame for face detection and drawing rectangles
        frame, face = process_frame(frame)

        # Update the deque with the most recent frame and face
        frame_deque.append(frame)
        face_deque.append(face)

    cap.release()
    print("Camera stopped!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_camera")
def start_camera_route():
    global camera_thread
    global camera_running

    if not camera_running:
        camera_thread = threading.Thread(target=start_camera)
        camera_thread.start()

    return "Camera started"


@app.route("/predict_emotion")
def predict_emotion():
    # print(len(frame_deque), len(face_deque))
    if len(frame_deque) > 0 and len(face_deque) > 0:
        # print("Frame not empty")
        face = face_deque[-1]  # Get the most recent face
        frame = frame_deque[-1]  # Get the most recent frame
        prediction = predict_emotion_from_frame(frame, face)
        return jsonify({"emotion": prediction})
    else:
        return jsonify({"emotion": "No face available"})


def generate_frames():
    while True:
        if len(frame_deque) > 0:
            frame = frame_deque[-1]  # Get the most recent frame
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
