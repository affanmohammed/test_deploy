import os
import cv2
import pickle
import pyttsx3 
import numpy as np
import streamlit as st
import mediapipe as mp
from matplotlib import pyplot as plt
#t2s = pyttsx3.init()

from tensorflow.keras.models import load_model
model = load_model('action.h5')



mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.array(['Hello','I am','Affan','Thanks', 'i love you','Fever','See you', 'God'])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                             )

sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
st.title("LSTM Sign Language ")
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # 3. Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            new_word = actions[np.argmax(res)]
                            #t2s.say(new_word)
                            #t2s.runAndWait()
                    else:
                        sentence.append(actions[np.argmax(res)])
                        new_word = actions[np.argmax(res)]
                        #t2s.say(new_word)
                        #t2s.runAndWait()

            if len(sentence) > 5:
                sentence = sentence[-5:]

        cv2.rectangle(image, (0, 0), (640, 40), (115,115,115), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the image in Streamlit
        frame_placeholder.image(image, channels="BGR")

        if stop_button_pressed:
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
