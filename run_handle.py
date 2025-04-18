import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.python.keras.models import load_model

# Create an HandLandmarker object.
Base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
HandLandmarker = mp.tasks.vision.HandLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions(base_options=Base_options,
                                                                    num_hands=2,
                                                                    running_mode=VisionRunningMode.VIDEO)

detector = HandLandmarker.create_from_options(HandLandmarkerOptions)

# Загрузка модели распознавателя жестов
#model = load_model('mp_hand_gesture')
 
# Все класссы жестов
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

path_video = '/home/v/download_youtube/yt-dlp/gestures.mp4'

cap = cv2.VideoCapture(path_video)
i=0
while cap.isOpened():
    _, frame = cap.read()
    w, h, c = frame.shape
    rgb_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord('q'):
                break
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image,timestamp_ms = i*4)
    i+=1
    print(result)
    
