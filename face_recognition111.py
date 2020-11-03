
import face_recognition
import cv2
import dlib
print(dlib.DLIB_USE_CUDA)
dlib.num
image_src = cv2.imread(r"1.png")
image = face_recognition.load_image_file("1.png")
print(image.shape)
w = image.shape[1]
h = image.shape[0]
import time
start = time.time()
face_locations = face_recognition.face_locations(image)
print(face_locations)
print((time.time() - start))

face_locations[0][1]-(face_locations[0][3]-face_locations[0][2]),
x=face_locations[0][3]
y=face_locations[0][0]

w=face_locations[0][1] - face_locations[0][3]
h=face_locations[0][2] - face_locations[0][0]
print(x)
print(y)


cv2.rectangle(image_src, (x,y), (x+w,y+h), (0, 0, 255), 2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image_src)
cv2.waitKey(0)

