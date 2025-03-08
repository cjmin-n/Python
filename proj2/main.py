#insightface/examples/demo_analysis.py

# STEP 1 : import modules
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2 : create inference module
app = FaceAnalysis() # 모델 자동으로 다운
app.prepare(ctx_id=0, det_size=(640,640)) 

# STEP 3 : load data
# img = ins_get_image('t1') # t1 이라는 샘플이미지 제공
img1 = cv2.imread('iu01.jpg')
img2 = cv2.imread('iu03.jpg')

# STEP 4 : inference
# faces = app.get(img) # 여러정보 뽑음.(위치, 나이, 성별 등)
# assert len(faces)==6 #에러체크
face1 = app.get(img1) 
# print(f"face1 : {face1}")
assert len(face1)==1

face2 = app.get(img2) 
assert len(face2)==1

# STEP 5 : post processing

# 5-1: draw face bounding box
# rimg = app.draw_on(img, faces) #draw_on 안에 deprecated된 변수가 있어서 오류날수있음 /draw_on 내부에서 np.int를 np.int_로 변경
# cv2.imwrite("./t1_output.jpg", rimg) # 이미지결과저장

# 5-2: calculate face similarity
# then print all-to-all face similarity

#normed_embedding 범위 큰 숫자들을 -1~1로 정규화
# feat1 = np.array(faces[0].normed_embedding , dtype=np.float32)
# feat2 = np.array(faces[1].normed_embedding , dtype=np.float32)
feat1 = np.array(face1[0].normed_embedding , dtype=np.float32)
feat2 = np.array(face2[0].normed_embedding , dtype=np.float32)
# print(f"feat1 : {feat1}")
sims = np.dot(feat1, feat2.T) #np.dot 행렬도?를 통해 두 배열의 유사도 측정하는 함수
print(sims) 
# 수지 단독 사진 2개 비교 : 0.5183684
# 얼마 이상이어야 같은 사람일까? 0.4이상

##마지막에러 pip install onnxruntime : onnxruntime 추론기 기반의 모델이기때문
