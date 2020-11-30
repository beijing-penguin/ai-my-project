import numpy as np
import cv2

def dis(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
img = cv2.imread('222.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,35,255,cv2.THRESH_BINARY)
dis(binary)

# #膨胀操作，因为是对线条进行提取定位，所以腐蚀可能会造成更大间隔的断点，将线条切断，因此仅做膨胀操作
# kernel = np.ones((20, 20), np.uint8)
# dilation = cv2.dilate(gray, kernel, iterations=1)

# kernel = np.ones((2, 2), np.uint8)
# dilation = cv2.dilate(binary, kernel, iterations=1)
# dis(dilation)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),3)
for i in range(0,len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 1)
    
dis(img)