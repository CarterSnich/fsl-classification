from flask import Flask, render_template, request
from classification import classify_sign
import numpy as np
import cv2 as cv
import base64

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    submitted_image = request.files["image"]
    in_memory_image = np.asarray(bytearray(submitted_image.read()), dtype=np.uint8)
    img = cv.imdecode(in_memory_image, cv.IMREAD_COLOR)

    result = classify_sign(img)

    if result:
        predicted_label, confidence, annotated_frame = result

        # Convert image to base64
        _, img_encoded = cv.imencode(".png", annotated_frame)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return {
            "prediction": predicted_label,
            "conifidence": confidence,
            "annotatedImage": img_base64,
        }
    else:
        return {
            "prediction": None,
            "conifidence": None,
            "annotatedImage": None,
        }
