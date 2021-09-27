import cv2
import numpy as np
from operator import itemgetter

# this scope directly comes from opencv docs
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# scope ends here

# name of images for iteration
imglist = ["im01", "im02", "im03", "im04", "im05", "im06", "im07", "im08", "im09", "im10", "im11", "im12", "im13",
           "im14", "im15", "im16", "im17", "im18", "im19", "im20"]

# suqar size
sS = 22
# put location of corners in the picture here
pointListPix = []
# put location of corners in the world here
# 6 and 9 are the number of row an column of corners
pointListLoc = np.zeros((9 * 6, 3), np.float32)
pointListLoc[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

worldPoints = []
imgPoints = []
coeMatrix = []

# for nameimg in imglist[0:11]:
# for nameimg in imglist[6:16]:
# for nameimg in imglist[10:]:
for nameimg in imglist:
    name = nameimg + ".jpg"
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # in this scope we fine corners with opencv functions
    # this scope directly comes from opencv docs
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    # scope ends here
    cornerimg = img

    # just reformating the corners list
    cornersN = []
    for i in corners2:
        cornersN.append((i[0, 0], i[0, 1]))
    # adding corners of current image to this list
    imgPoints.append(corners2)
    # adding corners location in real world of current image to this list
    worldPoints.append(pointListLoc)

# calculating value of calibration matrix
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(worldPoints, imgPoints, gray.shape[::-1], None, None)
print("focal x: ", mtx[0, 0])
print("focal y: ", mtx[1, 1])
print("pp x: ", mtx[0, 2])
print("pp y: ", mtx[1, 2])
