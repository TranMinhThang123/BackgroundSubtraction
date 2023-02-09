import numpy as np
import os
import cv2
import time


def VIBE(frame, samples, hashMin, N, R, phi, listSegMap):
    height, width = frame.shape
    # (480,640)

    segMap = np.array([[[255]] * width] * height)[:, :, 0]

    foregroundMatchCount = np.zeros((height, width)).astype(np.uint8)

    # classification
    clonesample = np.array([[[1000] * width] * height]*N)
    # print(clonesample.shape)
    res = np.zeros((height, width, N)).astype(np.uint8)
    block_frame = np.array([frame[:, :]] * N)
    posbg = np.where(listSegMap == 0)
    clonesample[posbg] = samples[posbg]
    res = abs(clonesample - block_frame)
    mask = res < R
    mask = np.sum(mask, axis=0)
    bgpos = np.where(mask >= hashMin)
    segMap[bgpos] = 0
    segMap = segMap.astype(np.uint8)

    #update
    samples[1:N, :, :] = samples[0:N-1, :, :]
    samples[0] = frame
    listSegMap[1:N, :, :] = listSegMap[0:N - 1, :, :]
    listSegMap[0] = segMap
    return segMap, samples, listSegMap


def initializeBackgroud(firstFrame, N):
    block_image = np.array([firstFrame] * N)

    paddedImage = np.pad(firstFrame, 1, 'symmetric')

    block_paddedImg = np.array([paddedImage[:, :]] * N)

    height, width = paddedImage.shape

    samples = np.zeros((N, height - 2, width - 2))

    noise_tensor1 = np.random.choice([-1, 0, 1], 480 * 640 * N)

    noise_tensor2 = np.random.choice([-1, 0, 1], 480 * 640 * N)

    noise_tensor3 = np.repeat(0, 480 * 640 * N)

    x = np.where(block_image == block_image)

    bufferx = x

    buffer1 = [np.ones(len(x[0]))] * 2

    buffer2 = np.zeros(len(x[0]))

    buffer1.insert(0, buffer2)

    buffer1 = np.array(buffer1)

    x = np.array(x, dtype=float)

    x += buffer1

    final_noise = np.array([noise_tensor3, noise_tensor2, noise_tensor1])

    x += final_noise

    x = np.array(x, dtype=int)

    x = tuple(x)

    print(block_paddedImg.shape)

    samples[bufferx] = block_paddedImg[x]
    # paddedImage = np.pad(firstFrame, 1, 'symmetric')
    #
    # height, width = paddedImage.shape
    #
    # samples = np.zeros((N, height, width))
    #
    # for k in range(N):
    #     for i in range(1, height - 1):
    #         for j in range(1, width - 1):
    #
    #             x, y = 0, 0
    #
    #             while (x == 0 and y == 0):
    #                 x = np.random.randint(-1, 1)
    #                 y = np.random.randint(-1, 1)
    #
    #             random_i = i + x
    #             random_j = j + y
    #
    #             samples[k, i, j] = paddedImage[random_i, random_j]
    #
    # samples = samples[:, 1:height - 1, 1:width - 1]

    width = 640
    height = 480
    segMap = np.array([[[255]] * width] * height)[:, :, 0]
    res = np.zeros((height, width, N)).astype(np.uint8)
    block_frame = np.array([firstFrame[:, :]] * N)
    res = block_frame - samples
    mask = res < R
    mask = np.sum(mask, axis=0)
    bgpos = np.where(mask >= hashMin)
    segMap[bgpos] = 0
    segMap = segMap.astype(np.uint8)
    listSegMap = np.array([segMap]*N)
    return samples, listSegMap


cap = cv2.VideoCapture(0)

success, firstFrame = cap.read()

firstGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)

N = 20
R = 20
hashMin = 2
phi = 5

samples, listSegMap = initializeBackgroud(firstGray, N)
print("Init done!!")
print("Sample shape = "+str(samples.shape))

while success:
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    time1 = time.time()
    segMap, samples, listSegMap = VIBE(gray, samples, hashMin, N, R, phi, listSegMap)
    # segMap = new_ViBE(gray, samples, hashMin, N, R, phi)
    time2 = time.time()
    print('Time updating: %.4f' % (time2-time1))
    cv2.imwrite("segMap.jpg", segMap)
    cv2.imwrite("Actual.jpg", frame)
    cv2.imwrite("Gray.jpg", gray)
    cv2.imshow('Actual Frame', frame)
    cv2.imshow('Gray Frame', gray)
    cv2.imshow('segMap Frame', segMap)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break while loop if video ends
        break

cv2.destroyAllWindows()