from typing import Union
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

import time
import cv2
import numpy as np

app = FastAPI()

# MediaPipe 모델 설정 
# - 모델은 속도를 위해 전역변수로 관리
base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite2.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)


# 이미지 시각화 설정
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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int = 1, q: Union[str, None] = "test"):
    return {"item_id": item_id, "q": q}

@app.post("/img_cls")
async def img_cls(
    image: UploadFile = File(...)
):
    start_time = time.time()
    # 이미지 파일 저장
    contents = await image.read()
    filename = f"temp_{image.filename}"
    with open(filename, "wb") as f:
        f.write(contents)

    # 이미지 로드 및 분류
    mp_image = mp.Image.create_from_file(filename)
    classification_result = classifier.classify(mp_image)

    # 결과 추출
    top_category = classification_result.classifications[0].categories[0]
    print(f"{top_category.category_name} ({top_category.score:.2f})")

    # 임시 파일 삭제
    os.remove(filename)
    end_time = time.time()
    print(f"처리 시간: {end_time - start_time:.2f}초")
    
    return JSONResponse(
        content={
            "message": "이미지 분류 요청이 성공적으로 처리되었습니다",
            "image_filename": image.filename,
            "top_category": top_category.category_name,
            "score": float(top_category.score)  # float32를 JSON 직렬화 가능한 형태로 변환
        },
        status_code=200
    )


@app.post("/obj_det")
async def obj_det(
    image: UploadFile = File(...)
):
    # 이미지 파일 저장
    contents = await image.read()
    filename = f"temp_{image.filename}"
    with open(filename, "wb") as f:
        f.write(contents)
    
    # 이미지 로드 및 감지
    mp_image = mp.Image.create_from_file(filename)
    detection_result = detector.detect(mp_image)

    # 결과 추출
    objects = []
    for detection in detection_result.detections:
        objects.append(detection)
    print(f"Find objects : {len(objects)}")

    # 시각화를 위한 이미지 복사
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite("test.jpg", rgb_annotated_image)
    return FileResponse("test.jpg") 






# @app.post("/face_recognition")
# async def face_recognition(
#     image1: UploadFile = File(...),
#     image2: UploadFile = File(...)
# ):
#     """
#     얼굴 인식 API
    
#     두 개의 이미지를 업로드하여 얼굴 인식을 수행합니다.
    
#     - **image1**: 첫 번째 이미지 파일
#     - **image2**: 두 번째 이미지 파일
    
#     Returns:
#         JSON 응답
#     """
#     # 여기에 얼굴 인식 로직을 구현할 수 있습니다
#     pass
    
#     return JSONResponse(
#         content={
#             "message": "얼굴 인식 요청이 성공적으로 처리되었습니다",
#             "image1_filename": image1.filename,
#             "image2_filename": image2.filename,
#         },
#         status_code=200
#     )