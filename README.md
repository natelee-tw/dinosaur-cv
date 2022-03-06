# dinosaur-cv
Using CV to play the Google Chrome Dinosaur Game

Another simple weekend project, using google's Posenet to play Chrome "Dinosaur Game".

## Authors
- Mengyong Lee | [LinkedIn](https://www.linkedin.com/in/mylee1/) | [Github](https://github.com/mylee16)
- Zi Yi Ewe | [LinkedIn](https://www.linkedin.com/in/zi-yi-ewe/)

## Method
Using Posenet, we detect key body points (keypoints) and find the points of the left and right eye. From there, we obtain the "y-axis" of the eye, and detect any "jump" by calculating the displacement of the y-axis between frames. We use pyautogui library to convert any "jump" signal from the algorithm to pressing the "up" button on the keyboard. 


## Instructions
- Clone this repo `https://github.com/mylee16/dinosaur-cv.git`
- `pip install -r requirements.txt` to install the required packages
- Run `python -m main`
- Go to google chrome, start the dinosaur game. The website we used for the dinosaur game: https://trex-runner.com/
- Click on the browser to start the game
- Start jumping and have fun!

Click below to see demo video :)
<a href="https://youtu.be/XzWKxh2am80" title="Playing Chrome Dinosaur Game with Computer Vision">
  <p align="center">
    <img width="75%" src="img/thumbnail.png" alt="Playing Chrome Dinosaur Game with Computer Vision"/>
  </p>
</a>

## Acknoledgement
Pose Estimation: https://www.tensorflow.org/lite/examples/pose_estimation/overview
Pyautogui: https://pyautogui.readthedocs.io/en/latest/