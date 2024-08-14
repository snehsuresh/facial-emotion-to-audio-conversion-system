# Facial Emotion to Audio Conversion System

This repository contains a real-time facial emotion detection system that converts detected emotions into audio feedback. It uses YOLO for emotion detection and various text-to-speech techniques to provide spoken feedback.

## Features

- Real-time facial emotion detection using YOLO.
- Audio feedback for detected emotions.
- Optimized for real-time performance.
- Configurable detection intervals and audio playback.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/facial-emotion-to-audio-conversion-system.git
    ```

2. **Navigate to the Project Directory**

    ```bash
    cd facial-emotion-to-audio-conversion-system
    ```

3. **Create and Activate a Virtual Environment**

    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

4. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Configure the necessary settings in `expressiondetection/config/model_config.py`:

 - `DETECTION_INTERVAL`: The time interval between consecutive detections.
 - API keys and other model configurations if applicable.

## Usage

 To start the real-time facial emotion detection:

 ```bash
 python app.py
 ```

 The application will open a video stream from your camera and display the detected emotion on the video feed. Audio feedback will be played based on the detected emotion.

 ## Dependencies

 - OpenCV
 - gTTS
 - pygame
 - other libraries listed in `requirements.txt`

## Troubleshooting

 - Ensure your camera is properly connected and accessible.
 - Make sure all dependencies are installed correctly.
 - Check for any error messages and refer to the respective library documentation for troubleshooting.

 ## Contributing

 Contributions are welcome! Please follow these steps to contribute:

 1. Fork the repository.
 2. Create a new branch for your changes.
 3. Make your changes and commit them.
 4. Push your changes to your fork.
 5. Open a pull request.

 ## License

 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 ## Contact

 If you have any questions or suggestions, feel free to reach out:

 - Email: snehsuresh02@gmail.com

 ## Acknowledgements

 - YOLO for emotion detection.
 - Expression Dataset by Fardhansyah Hanafi, https://universe.roboflow.com/fardhansyah-hanafi-d9mrp/expression-bivfq 


Thank you for using the Facial Emotion to Audio Conversion System!
