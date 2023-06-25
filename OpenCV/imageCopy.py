
import cv2
import numpy
import numpy as np

imageFile = "IMG_1134.JPG"

img = cv2.imread(imageFile)
(h, w)=img.shape[:2]
# 查找mask
# 图片裁剪 [start_row:end_row, start_col:end_col]
# 开始行
start_row = 0
# 结束行
end_row = 500
# 开始列
start_col = 0
# 结束列
end_col = 500
img1 = img[start_row:end_row, start_col:end_col]
print(img1)
# 合并图像
img[start_row:end_row, start_col:end_col] = img1
cv2.imshow("mask applied to image", img)
cv2.waitKey(0)
