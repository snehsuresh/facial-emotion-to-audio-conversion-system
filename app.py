from flask import Flask, render_template, Response, request, jsonify
import collections
import cv2
import numpy as np
from expressiondetection.live.video_stream import (
    process_frame,
    predict_emotion_from_frame,
)

app = Flask(__name__)

frame_deque = collections.deque(maxlen=1)  # Only keep the most recent frame
face_deque = collections.deque(maxlen=1)  # Only keep the most recent face


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame_route():
    if "video_frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    video_frame = request.files["video_frame"]
    frame = np.asarray(bytearray(video_frame.read()), dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Process frame for face detection and drawing rectangles
    frame, face = process_frame(frame)

    # Update the deque with the most recent frame and face
    frame_deque.append(frame)
    face_deque.append(face)

    # Send the processed frame back to the client
    _, buffer = cv2.imencode(".jpg", frame)
    frame_bytes = buffer.tobytes()

    return Response(frame_bytes, mimetype="image/jpeg")


@app.route("/predict_emotion", methods=["POST"])
def predict_emotion_route():
    if len(frame_deque) == 0 or len(face_deque) == 0:
        return jsonify({"emotion": "No face available"})

    face = face_deque[-1]  # Get the most recent face
    frame = frame_deque[-1]  # Get the most recent frame
    prediction = predict_emotion_from_frame(frame, face)

    return jsonify({"emotion": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
