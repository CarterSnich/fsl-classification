import cv2 as cv
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from decimal import Decimal

classes = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

model = load_model("latest.h5")


# Function to extract features from an input image using MediaPipe Hands
def extract_feature(input_frame):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1
    ) as hands:
        # Flip the frame for accurate processing
        flipped_frame = cv.flip(input_frame, 1)
        results = hands.process(cv.cvtColor(flipped_frame, cv.COLOR_BGR2RGB))

        # Get frame dimensions
        image_height, image_width, _ = input_frame.shape

        if not results.multi_hand_landmarks:
            # Return zero landmarks if no hands are detected
            return False

        # Extract landmarks for the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            z = landmark.z
            landmarks.extend([x, y, z])

        # Annotate the image with hand landmarks and connections
        annotated_frame = flipped_frame.copy()
        mp_drawing.draw_landmarks(
            annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        return landmarks, annotated_frame


# Function to create a structured input array
def create_input_array(landmarks):
    input_IMG = np.array(
        [
            [
                [landmarks[0]],
                [landmarks[1]],
                [landmarks[2]],  # wrist
                [landmarks[3]],
                [landmarks[4]],
                [landmarks[5]],  # thumb_Cmc
                [landmarks[6]],
                [landmarks[7]],
                [landmarks[8]],  # thumb_Mcp
                [landmarks[9]],
                [landmarks[10]],
                [landmarks[11]],  # thumb_Ip
                [landmarks[12]],
                [landmarks[13]],
                [landmarks[14]],  # thumb_Tip
                [landmarks[15]],
                [landmarks[16]],
                [landmarks[17]],  # index_Mcp
                [landmarks[18]],
                [landmarks[19]],
                [landmarks[20]],  # index_Pip
                [landmarks[21]],
                [landmarks[22]],
                [landmarks[23]],  # index_Dip
                [landmarks[24]],
                [landmarks[25]],
                [landmarks[26]],  # index_Tip
                [landmarks[27]],
                [landmarks[28]],
                [landmarks[29]],  # middle_Mcp
                [landmarks[30]],
                [landmarks[31]],
                [landmarks[32]],  # middle_Pip
                [landmarks[33]],
                [landmarks[34]],
                [landmarks[35]],  # middle_Dip
                [landmarks[36]],
                [landmarks[37]],
                [landmarks[38]],  # middle_Tip
                [landmarks[39]],
                [landmarks[40]],
                [landmarks[41]],  # ring_Mcp
                [landmarks[42]],
                [landmarks[43]],
                [landmarks[44]],  # ring_Pip
                [landmarks[45]],
                [landmarks[46]],
                [landmarks[47]],  # ring_Dip
                [landmarks[48]],
                [landmarks[49]],
                [landmarks[50]],  # ring_Tip
                [landmarks[51]],
                [landmarks[52]],
                [landmarks[53]],  # pinky_Mcp
                [landmarks[54]],
                [landmarks[55]],
                [landmarks[56]],  # pinky_Pip
                [landmarks[57]],
                [landmarks[58]],
                [landmarks[59]],  # pinky_Dip
                [landmarks[60]],
                [landmarks[61]],
                [landmarks[62]],  # pinky_Tip
            ]
        ]
    )
    return input_IMG


def classify_sign(image):
    # Extract hand features and annotated frame
    # frame = imutils.resize(frame, width=640)
    result = extract_feature(image)

    if result:  # Ensure landmarks were detected
        features, annotated_frame = result
        input_IMG = create_input_array(features)

        # Reshape input for prediction
        input_IMG = input_IMG.reshape(1, -1)  # Reshape to (1, 63) if necessary

        # Model prediction
        prediction = model.predict(input_IMG)
        predictions = np.argmax(prediction, axis=1)
        confidence = prediction[0][predictions[0]]

        confidence_decimal = Decimal(float(confidence))
        percentage = confidence_decimal * 100

        predicted_label = classes[predictions[0]]

        return predicted_label, f"{percentage:.2f}%", annotated_frame

    else:
        return False
