import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
### 首先将图片转化为灰度图像
image = cv2.imread("222.jpg")
def rgb2gray(image):
    h = image.shape[0]
    w = image.shape[1]
    grayimage  = np.zeros((h,w),np.uint8)
    for i in tqdm(range(h)):
        for j in range(w):
            grayimage [i,j] = 0.144*image[i,j,0]+0.587*image[i,j,1]+0.299*image[i,j,1]
    return grayimage

### 大津法
def otsu(image):
    ### 高和宽
    h = image.shape[0]
    w = image.shape[1]
    ### 求总像素
    m = h*w

    otsuimg = np.zeros((h, w), np.uint8)
    ##初始阈值
    initial_threshold = 0
    ### 最终阈值
    final_threshold   = 0
    # 初始化各灰度级个数统计参数
    histogram = np.zeros(256, np.int32)
    # 初始化各灰度级占图像中的分布的统计参数
    probability = np.zeros(256, np.float32)

    ### 各个灰度级的个数统计
    for i in tqdm(range(h)):
        for j in range(w):
            s = image[i,j]
            histogram[s] = histogram[s] +1
    ### 各灰度级占图像中的分布的统计参数
    for i in tqdm(range(256)):
        probability[i] = histogram[i]/m

    for i in tqdm(range(255)):
        w0 = w1 = 0  ## 前景和背景的灰度数
        fgs = bgs = 0  # 定义前景像素点灰度级总和背景像素点灰度级总和
        for j in range(256):
            if j <= i:  # 当前i为分割阈值
                w0 += probability[j]  # 前景像素点占整幅图像的比例累加
                fgs += j * probability[j]
            else:
                w1 += probability[j]  # 背景像素点占整幅图像的比例累加
                bgs += j * probability[j]
        u0 = fgs / w0  # 前景像素点的平均灰度
        u1 = bgs / w1  # 背景像素点的平均灰度
        G  = w0*w1*(u0-u1)**2
        if G >= initial_threshold:
            initial_threshold = G
            final_threshold = i
    print(final_threshold)

    for i in range(h):
        for j in range(w):
            if image[i, j] > final_threshold:
                otsuimg[i, j] = 255
            else:
                otsuimg[i, j] = 0
    return otsuimg

grayimage  = rgb2gray(image)
otsuimage  = otsu(grayimage)

cv2.imshow("grayimage",grayimage)
cv2.imshow("otsuimage",otsuimage)

cv2.waitKey()
# print(new_image)