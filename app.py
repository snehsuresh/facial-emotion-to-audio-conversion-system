from flask import Flask, render_template, Response, jsonify
from expressiondetection.live.video_stream import predict_live
import threading
import queue
import cv2

app = Flask(__name__)

# Variables for camera
camera_thread = None
camera_running = False
predicting = False
frame_queue = queue.Queue(maxsize=10)


def start_camera():
    global camera_running
    if not camera_running:
        camera_running = True
        # Start capturing from the camera
        cap = cv2.VideoCapture(0)
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)
        cap.release()


def start_prediction():
    global predicting
    if not predicting:
        predicting = True
        predict_live(frame_queue)


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


@app.route("/stop_camera")
def stop_camera_route():
    global camera_running
    camera_running = False
    return "Camera stopped"


@app.route("/start_prediction")
def start_prediction_route():
    global predicting
    if not predicting:
        threading.Thread(target=start_prediction).start()
    return "Prediction started"


def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
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
