import cv2
from PIL import Image

# from gtts import gTTS
import cv2

# from gtts import gTTS
# from datetime import datetime
# from flask import Response
# import io

label_translation = {
    "Jijik": "Disgusted",
    "Kaget": "Surprised",
    "Marah": "Angry",
    "Sedih": "Sad",
    "Senang": "Happy",
    "Takut": "Fearful",
    "Tidak Berekspresi": "No Expression",
}


def translate_label(indonesian_label):
    return label_translation.get(indonesian_label, "Unknown Label")


def display_image(image):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_pil.show()


# def speak_text(
#     text,
# ):
#     tts = gTTS(text, lang="en")
#     audio_stream = io.BytesIO()
#     tts.write_to_fp(audio_stream)
#     audio_stream.seek(0)

#     # Return audio stream as response with appropriate headers
#     return Response(audio_stream, mimetype="audio/mpeg")


# def speak_text(text):
#     mp3_fp = BytesIO()
#     tts = gTTS(text=text, lang="en")
#     # tts.save("output.mp3")
#     tts.write_to_fp(mp3_fp)
#     sound = mp3_fp
#     sound.seek(0)
#     mixer.init()

#     mixer.music.load(sound, "mp3")
#     mixer.music.play()
#     # time.sleep(5)
# os.system("afplay output.mp3")
