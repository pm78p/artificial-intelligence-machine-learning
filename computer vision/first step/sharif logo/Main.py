import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def autocrop(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

someparameter = 161
someparameter *= 5

img = cv2.imread("logo.PNG", cv2.IMREAD_COLOR)
d = 25
f = 500
picturesize = 256
px1 = picturesize / 2
py1 = picturesize / 2
px2 = 0
py2 = 0
theta = math.acos(25 / 40)
dist2centers = math.sqrt(40 ** 2 - 25 ** 2)
t = np.array([dist2centers * math.sin(theta), 0, dist2centers * math.cos(theta)])
n = np.array([-math.sin(theta), 0, math.cos(theta)])
px2 += px1 + t[0]
py2 += (picturesize + someparameter) / 2

cT = math.cos(theta)
sT = math.sin(theta)
R = np.array([[cT, 0, sT], [0, 1, 0], [-sT, 0, cT]])

k = np.array([[f, 0, px1], [0, f, py1], [0, 0, 1]])
k2 = np.array([[f, 0, px2], [0, f, py2], [0, 0, 1]])

middle = R - np.dot(t, n.T) / d

H = np.dot(k2, middle)
H = np.dot(H, np.linalg.inv(np.matrix(k)))
print(H)

im_dst = cv2.warpPerspective(img, H, (picturesize + someparameter, picturesize + someparameter))

cv2.imwrite("r12.jpg", autocrop(im_dst))
