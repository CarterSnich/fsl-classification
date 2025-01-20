from flask import Flask, render_template, request
from classification import classify_sign
import numpy as np
import cv2 as cv
import base64

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = None
app.config["MAX_FORM_MEMORY_SIZE"] = 50 * (2**10) ** 2


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    base64_image = request.form["image"]

    image_data = base64.b64decode(base64_image)
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv.imdecode(np_array, cv.IMREAD_COLOR)

    result = classify_sign(img)

    if result:
        predicted_label, confidence, annotated_frame = result

        # Convert image to base64
        _, img_encoded = cv.imencode(".png", annotated_frame)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            # "annotatedImage": img_base64,
        }
    else:
        return {
            "prediction": None,
            "confidence": None,
            # "annotatedImage": None,
        }
