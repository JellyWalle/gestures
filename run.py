import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.gestures
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]
    
      # Draw the hand landmarks.
      hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())
    
      # Get the top left corner of the detected hand's bounding box.
      height, width, _ = annotated_image.shape
      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]
      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - MARGIN
    
      # Draw handedness (left or right hand) on the image.
      cv2.putText(annotated_image, f"{handedness[0].category_name}",
                  (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image

# Create an HandLandmarker object.
Base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
GestureRecognizer = vision.GestureRecognizer
VisionRunningMode =vision.RunningMode
GestureRecognizerResult = vision.GestureRecognizerResult
RESULT = None

def get_result(result:GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global RESULT
    RESULT = result

GestureRecognizerOptions = vision.GestureRecognizerOptions(base_options=Base_options,
                                                                    running_mode=VisionRunningMode.LIVE_STREAM,
                                                                    num_hands=2,
                                                                    result_callback=get_result)

detector = GestureRecognizer.create_from_options(GestureRecognizerOptions)

cap = cv2.VideoCapture(0)
timestamp=0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detector.recognize_async(mp_image, timestamp)
    
    if RESULT is not None:
        frame = draw_landmarks_on_image(frame,RESULT)
    timestamp+=1
    
    cv2.imshow("Live", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
