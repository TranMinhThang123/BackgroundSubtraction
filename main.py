import numpy as np
import os
import cv2
from ViBeAlgo import vibe_gray
import  time
# ViBe = vibe_gray()
#
# cap = cv2.VideoCapture(0)
#
# success, firstFrame = cap.read()
#
# if success:
#     cv2.imwrite("ff.jpg", firstFrame)
#
# firstFrame = cv2.imread('ff.jpg', 0)
#
# if success:
#     ViBe.AllocInit(firstFrame)
#
#
# while True:
#     success, frame = cap.read()
#     cv2.imwrite("ff.jpg", frame)
#     frame = cv2.imread('ff.jpg', 0)
#     segMap = ViBe.Segmentation(frame)
#     cv2.imshow('segMap', segMap)
#
#
#     if cv2.waitKey(1) and 0xff == ord('q'):
#         break
#
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
vibe = vibe_gray()

frame_index = 0
segmentation_time = 0
update_time = 0
t1 = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_index % 100 == 0:
        print('Frame number: %d' % frame_index)

    if frame_index == 0:
        vibe.AllocInit(gray_frame)

    t2 = time.time()
    segmentation_map = vibe.Segmentation(gray_frame)
    t3 = time.time()
    vibe.Update(gray_frame, segmentation_map)
    t4 = time.time()
    segmentation_time += (t3 - t2)
    update_time += (t4 - t3)
    print('Frame %d, segmentation: %.4f, updating: %.4f' % (frame_index, t3 - t2, t4 - t3))
    # segmentation_map = cv2.medianBlur(segmentation_map, 3)

    cv2.imshow('Actual Frame!', frame)
    cv2.imshow('Gray Frame!', gray_frame)
    cv2.imshow('Segmentation Frame!', segmentation_map)
    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break while loop if video ends
        break
t5 = time.time()
print('All time cost %.3f' % (t5 - t1))
print('segmentation time cost: %.3f, update time cost: %.3f' % (segmentation_time, update_time))

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()