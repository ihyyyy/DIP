import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

input = io.imread('./ycz_inverted.jpg')
preds = fa.get_landmarks(input)


import cv2
import numpy as np

# 检查是否检测到人脸关键点
if preds is not None:
    # 将skimage图像转换为OpenCV格式
    img = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
    
    # 遍历每个检测到的人脸
    for facial_landmarks in preds:
        # 绘制68个关键点
        for (x, y) in facial_landmarks:
            cv2.circle(img, (int(x), int(y)), 1, (0,0,255), -1)
    
    # 保存图像
    # cv2.imwrite('ycz_inverted_with_landmarks.jpg', img)
else:
    print("未检测到人脸关键点")
    
# save the array preds
np.save('preds.npy', preds)
