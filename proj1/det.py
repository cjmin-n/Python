# object_detector 객체감지

# 1.Visualization utilities
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

# 2. 이미지 다른이름으로 저장 cat_and_dog.jpg

# 이미지 띄움
# import cv2

# img = cv2.imread('cat_and_dog.jpg') # 메모리상의 이미지 가져옴
# cv2.imshow("test", img) #이름(test)을 정해줘야함
# cv2.waitKey(0) #바로 꺼지지 않게 하기위해서 키를 누를때까지 기다림


# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
# base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite2.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5) #score_threshold 확률값이 0.5이상인것만 가져오기(0.5가 기준)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
# image = mp.Image.create_from_file('cat_and_dog.jpg')
image = mp.Image.create_from_file('persons.jpg')

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)
# print(detection_result)

# 1. 전신이 나와있는 사람 사진을 인터넷에서 다운 받을 것 (jpg, jpeg)
# 2. 사진속에 사람이 있으며 몇명이 있는지 출력하시오. ( print(count) )
# 내가 푼 방식
# count = 0
# for detection in detection_result.detections:
#     for category in detection.categories:
#         if category.category_name == 'person':
#             count+=1

# print(count) #3

# 쌤이 알려준 방식
persons = []
for detection in detection_result.detections:
   if detection.categories[0].category_name == 'person':
      persons.append(detection)
print(f"Find Person : {len(persons)}")

# STEP 5: Process the detection result. In this case, visualize it.
# 기존 버전
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) # bgr->rgb로 변환

cv2.imshow("test", rgb_annotated_image)
cv2.waitKey(0)



