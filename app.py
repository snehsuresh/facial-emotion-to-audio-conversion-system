from flask import Flask, render_template, Response
from expressiondetection.live.video_stream import predict_live
import threading
import queue
import cv2

app = Flask(__name__)

# Variables for camera
camera_thread = None
camera_running = False
frame_queue = queue.Queue(maxsize=10)  # Set a reasonable max size for the queue


def start_camera():
    global camera_running
    if not camera_running:
        camera_running = True
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
    app.run(debug=True)
