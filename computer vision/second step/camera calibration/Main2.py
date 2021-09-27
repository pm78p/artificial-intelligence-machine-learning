import cv2
import numpy as np
from operator import itemgetter
import math
import random


# in this function we find a vanishing point
# first we calculate parallel lines by 2 points
# after that we find one intersect for each pair of lines
# and in the end avereging those intersections as a vanishing point
def findVanishing(points):
    lineList = []
    intersectList = []
    # here just calculatin lines
    for i in points:
        l = findLine(i[0], i[1])
        lineList.append(l)
    # here finding intersections
    for i in lineList:
        for j in lineList:
            if i[0] != j[0] or i[1] != j[1] or i[2] != j[2]:
                xep = (j[2] * i[1] - i[2] * j[1]) / (j[1] * i[0] - i[1] * j[0])
                yep = (j[2] * i[0] - i[2] * j[0]) / (j[0] * i[1] - i[0] * j[1])
                intersectList.append((xep, yep))
    # and get average here as vanishing point
    totalx = 0
    totaly = 0
    for i in intersectList:
        totalx += i[0]
        totaly += i[1]

    return (totalx / len(intersectList), totaly / len(intersectList))


# this function find Coefficient matrix of line which passed two points
def findLine(p1, p2):
    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    c = p1[1] - a * p1[0]
    return (a, -1, c)


# same as Main file
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# same as Main file
imglist = ["im01", "im02", "im03", "im04", "im05", "im06", "im07", "im08", "im09", "im10", "im11", "im12", "im13",
           "im14", "im15", "im16", "im17", "im18", "im19", "im20"]

# same as Main file
pointListPix = []

# it just sum the focal length to calculate an average
totalfocal = 0
# in this list we add focal length values
focalList = []
# loop to find focal lengh in every picturs
for nameimg in imglist:
    name = nameimg + ".jpg"
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # pricipal points of picture based on similarity of camera and plane centers
    px = np.shape(img)[1]
    py = np.shape(img)[0]

    # repitetive things and finding corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    #just reformatin corners list
    cornersN = []
    for i in corners2:
        cornersN.append((i[0, 0], i[0, 1]))

    #here we get 2 points facing each other around the chessboard
    listPointsLinesCol = []
    for i in range(0, 9):
        listPointsLinesCol.append((cornersN[i], cornersN[i + 9 * 5]))


    listPointsLinesWid = []
    for i in range(0, 46, 9):
        listPointsLinesWid.append((cornersN[i], cornersN[i + 8]))

    v1 = findVanishing(listPointsLinesWid)
    v1 = (v1[0], v1[1], 1)
    v2 = findVanishing(listPointsLinesCol)
    v2 = (v2[0], v2[1], 1)
    focalLenghtPt1 = v2[0] * (v1[0] - v1[2] * px)
    focalLenghtPt2 = v2[1] * (v1[1] - v1[2] * py)
    focalLenghtPt3 = v2[2] * (v1[2] * (px ** 2 + py ** 2) - v1[0] * px - v1[1] * py)
    focalLenght = math.sqrt(math.fabs((focalLenghtPt1 + focalLenghtPt2 + focalLenghtPt3) / ((-1) * v1[2] * v2[2])))
    focalList.append(focalLenght)
    totalfocal += focalLenght

avg = totalfocal / len(imglist)
deleter = 0
focaltotal = 0
for i in focalList:
    if i > avg:
        deleter += 1
    else:
        print(i)
        focaltotal += i

print(focaltotal/(len(imglist) - deleter))