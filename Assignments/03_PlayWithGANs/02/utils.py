import numpy as np
import cv2
from skimage import io

image = io.imread('./ycz_inverted.jpg')
preds = np.load('preds.npy')


img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 创建不同颜色的点来表示不同部位
# colors = {
#     'jaw': landmarks[0:17],    # 下巴
#     'left_eyebrow': landmarks[17:22],    # 左眉毛
#     'right_eyebrow': landmarks[22:27],   # 右眉毛
#     'nose_bridge': landmarks[27:31],     # 鼻梁
#     'nose_tip': landmarks[31:36],        # 鼻尖
#     'left_eye': landmarks[36:42],        # 左眼
#     'right_eye': landmarks[42:48],       # 右眼
#     'outer_lip': landmarks[48:60],       # 外嘴唇
#     'inner_lip': landmarks[60:68]        # 内嘴唇
# }

for facial_landmarks in preds:
    for (x, y) in facial_landmarks[48:60]:
        cv2.circle(img, (int(x), int(y)), 1, (0,0,255), -1)
x,y = preds[0,48,:]
cv2.circle(img, (int(x), int(y)), 1, (255,0,0), -1) # 左嘴角 blue
x,y = preds[0,54,:]
cv2.circle(img, (int(x), int(y)), 1, (0,255,0), -1) # 右嘴角 green
x,y = preds[0,57,:]
cv2.circle(img, (int(x), int(y)), 1, (0,255,255), -1) # 下嘴唇 yellow

cv2.imwrite('ycz_inverted_with_landmarks.jpg', img)