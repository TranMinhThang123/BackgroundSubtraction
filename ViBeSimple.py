import numpy as np
import os
import cv2


def VIBE(frame, samples, hashMin, N, R, phi):
    height, width = frame.shape
    #(480,640)

    segMap = np.array([[[255]]*width]*height)[:, :, 0]

    foregroundMatchCount = np.zeros((height, width)).astype(np.uint8)

    #classification
    res = np.zeros((height, width, N)).astype(np.uint8)
    block_frame = np.array([frame[:, :]] * N)
    res = block_frame-samples
    mask = res < R
    mask = np.sum(mask, axis=0)
    bgpos = np.where(mask >= hashMin)
    segMap[bgpos] = 0
    segMap = segMap.astype(np.uint8)

    #update
    randomInBackground = np.random.randint(0, 16, size=(height, width)) #tao ra 1 mang random cac gia tri tu 1->16 kich thuoc = kich thuoc cua frame
    bufferSeg = np.array(randomInBackground == segMap)
    bufferSeg.astype(np.uint8)
    updatePos = np.where(bufferSeg == 0)
    randomImg = np.random.randint(1, 16, len(updatePos[0]))
    bufferPos = list(updatePos)
    bufferPos.insert(0, randomImg)#gan cac vi tri trong samples vao mang cac vi tri duoc update
    updatePos = tuple(bufferPos)
    samples[updatePos] = block_frame[updatePos]
    #update neighboor
    randomInBackground = np.random.randint(0, 20, size=(height, width))  # tao ra 1 mang random cac gia tri tu 1->16 kich thuoc = kich thuoc cua frame
    bufferSeg = np.array(randomInBackground == segMap)
    indexes = np.random.randint(-1, 2, size=(bufferSeg.shape))
    bufferSeg = bufferSeg+indexes
    bufferSeg.astype(np.uint8)
    updatePos = np.where(bufferSeg == 0)
    randomImg = np.random.randint(0, 20, len(updatePos[0]))
    bufferPos = list(updatePos)
    bufferPos.insert(0, randomImg)  # gan cac vi tri trong samples vao mang cac vi tri duoc update
    updatePos = tuple(bufferPos)
    samples[updatePos] = block_frame[updatePos]
    # r = np.random.randint(0, N)
    samples[1:N, :, :] = samples[0:N-1, :, :]
    samples[0] = frame
    return segMap, samples


def initializeBackgroud(firstFrame, N):

    block_image = np.array([firstFrame]*N)

    paddedImage = np.pad(firstFrame, 1, 'symmetric')

    block_paddedImg = np.array([paddedImage[:, :]]*N)

    height, width = paddedImage.shape

    samples = np.zeros((N, height-2, width-2))

    noise_tensor1 = np.random.choice([-1, 0, 1], 480 * 640*N)

    noise_tensor2 = np.random.choice([-1, 0, 1], 480 * 640*N)

    noise_tensor3 = np.repeat(0, 480*640*N)

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

    return samples


cap = cv2.VideoCapture(0)

success, firstFrame = cap.read()

firstGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)

N = 35
R = 20
hashMin = 17
phi = 5

samples = initializeBackgroud(firstGray, N)
print("Init done!!")
print("Sample shape = "+str(samples.shape))

while success:
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    segMap, samples = VIBE(gray, samples, hashMin, N, R, phi)
    # segMap = new_ViBE(gray, samples, hashMin, N, R, phi)
    cv2.imwrite("ff.jpg", segMap)
    cv2.imshow('actual frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('segMap', segMap)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break while loop if video ends
        break

cv2.destroyAllWindows()