import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Configs
threshold = 0.5

# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures["serving_default"]

# Using Webcam
capture = cv2.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    frame = frame[:, 280:1000, :]
    frame = cv2.flip(frame, 1)

    # Load the input image.

    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_processed = tf.expand_dims(frame_processed, axis=0)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    frame_processed = tf.cast(tf.image.resize_with_pad(frame_processed, 192, 192), dtype=tf.int32)

    # Run model inference.
    outputs = movenet(frame_processed[:,:,:,:3])
    keypoints = outputs["output_0"]

    print(keypoints)
    keypoints = keypoints[keypoints[:,:,:,2] > threshold]

    for keypoint in keypoints:
        point_x = keypoint[1].numpy() * frame.shape[1]
        point_y = keypoint[0].numpy() * frame.shape[0]
        print(point_x, point_y)

        frame = cv2.drawMarker(frame, (round(point_x), round(point_y)), (0,0,255), markerType=cv2.MARKER_STAR, 
        markerSize=20, thickness=2, line_type=cv2.LINE_AA)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        # Quit when q is pressed
        break

capture.release()
cv2.destroyAllWindows()
