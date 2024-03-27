import cv2
import mediapipe as mp
import matplotlib as plt
# 初始化MediaPipe的人脸检测组件
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# 读取图像
image = cv2.imread('img/1.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行人脸检测
results = face_detection.process(gray_image)

# 绘制人脸检测结果框
if results.detections:
    for detection in results.detections:
        x, y, width, height = detection.location_data.relative_bounding_box
        top, bottom = max(0, y - height // 4), min(image.shape[0], y + height // 4)
        left, right = max(0, x - width // 4), min(image.shape[1], x + width // 4)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# 显示结果图像
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()