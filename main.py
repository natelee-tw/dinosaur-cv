import cv2
import pyautogui
import tensorflow as tf
import tensorflow_hub as hub

from enum import Enum, auto
from collections import deque

# Configs
confidence_threshold = 0.3
min_jump_disp = 20
frame_reload_counter = 3

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

counter = 0

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
    keypoints = keypoints[keypoints[:,2] > confidence_threshold] # check threshold

    sum_y, num_y = 0, 0
    for keypoint in keypoints:
        point_x = keypoint[1].numpy() * frame.shape[1]
        point_y = keypoint[0].numpy() * frame.shape[0]
        sum_y += point_y
        num_y += 1

        frame = cv2.drawMarker(frame, (round(point_x), round(point_y)), (0,0,255), markerType=cv2.MARKER_STAR, 
        markerSize=10, thickness=1, line_type=cv2.LINE_AA)
    avg_y = int(sum_y / num_y) if num_y != 0 else 0

    frame_queue.append(avg_y)
    frame_disp = int(frame_queue[-1] - frame_queue[0])

    if (frame_disp < -min_jump_disp) and counter > frame_reload_counter:
        counter = 0 # skip frames after pressing jump
        pyautogui.press('up')
        action="jump"
        print("jump")

    else:
        counter += 1
        action="do nothing"

    if counter < frame_reload_counter:
        cv2.putText(frame, "jump", (200, 400), cv2.FONT_HERSHEY_PLAIN, 
                    2.3, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "do nothing", (200, 400), cv2.FONT_HERSHEY_PLAIN, 
                    2.3, (0, 255, 0), 2, cv2.LINE_AA) 

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()