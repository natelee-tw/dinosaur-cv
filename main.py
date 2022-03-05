import cv2
import numpy as np
import pyautogui
import tensorflow as tf
import tensorflow_hub as hub

from enum import Enum, auto
from collections import deque

# Configs
threshold = 0.5

# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures["serving_default"]

# Using Webcam
capture = cv2.VideoCapture(0)
frame_queue = deque([], maxlen=2)


class Action(Enum):
    JUMP = auto()
    DUCK = auto()
    NOTHING = auto()


while True:
    isTrue, frame = capture.read()
    frame = frame[:, 280:1000, :]
    frame = cv2.flip(frame, 1)

    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_processed = tf.expand_dims(frame_processed, axis=0)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    frame_processed = tf.cast(tf.image.resize_with_pad(frame_processed, 192, 192), dtype=tf.int32)

    # Run model inference.
    outputs = movenet(frame_processed[:,:,:,:3])
    keypoints = outputs["output_0"]

    keypoints = keypoints[0][0][1:3] # take only left and right eye
    keypoints = keypoints[keypoints[:,2] > threshold] # check threshold

    sum_y, num_y= 0, 0
    for keypoint in keypoints:
        point_x = keypoint[1].numpy() * frame.shape[1]
        point_y = keypoint[0].numpy() * frame.shape[0]
        sum_y += point_y
        num_y += 1

        frame = cv2.drawMarker(frame, (round(point_x), round(point_y)), (0,0,255), markerType=cv2.MARKER_STAR, 
        markerSize=20, thickness=2, line_type=cv2.LINE_AA)
    avg_y = sum_y / num_y if num_y != 0 else 0

    frame_queue.append(avg_y)
    frame_dist = frame_queue[-1] - frame_queue[0]
    if frame_dist > 20:
        pyautogui.press('up')
        action="jump"
        print("jump")

    else:
        action="do nothing"


    cv2.putText(frame, str(int(frame_dist)), (200, 200), cv2.FONT_HERSHEY_PLAIN, 
                2.3, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, action, (200, 400), cv2.FONT_HERSHEY_PLAIN, 
                2.3, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        # Quit when q is pressed
        break

capture.release()
cv2.destroyAllWindows()