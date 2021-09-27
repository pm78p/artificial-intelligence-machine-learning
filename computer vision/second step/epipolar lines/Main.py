import cv2
import numpy as np
import scipy
from scipy.linalg import null_space
import random


def findEpipole(f, points):
    lineList = []
    epipoList = []
    for i in points:
        l = np.dot(f, np.array([i[0], i[1], 1]))
        lineList.append(l)
    for i in lineList:
        for j in lineList:
            if i[0] != j[0] and i[1] != j[1] and i[2] != j[2]:
                xep = (j[2] * i[1] - i[2] * j[1]) / (j[1] * i[0] - i[1] * j[0])
                yep = (j[2] * i[0] - i[2] * j[0]) / (j[0] * i[1] - i[0] * j[1])
                epipoList.append((xep, yep))
    totalx = 0
    totaly = 0
    for i in epipoList:
        totalx += i[0]
        totaly += i[1]

    return (totalx / len(epipoList), totaly / len(epipoList))


name1 = "Building1Resized"
name2 = "Building2Resized"
img1 = cv2.imread(name1 + ".jpg")
img1g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(name2 + ".jpg")
img2g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift2 = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift2.detectAndCompute(img1g, None)
kp2, des2 = sift2.detectAndCompute(img2g, None)

match = cv2.BFMatcher()
matches = match.knnMatch(des1, des2, k=2)

src_pts = []
dst_pts = []
matches1to2 = []
for i, j in matches:
    if i.distance / j.distance < 6.5 / 10:
        matches1to2.append(i)

src_pts = np.float64([kp1[m.queryIdx].pt for m in matches1to2]).reshape(-1, 1, 2)
dst_pts = np.float64([kp2[m.trainIdx].pt for m in matches1to2]).reshape(-1, 1, 2)
print(len(src_pts))
F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)
src_ptsN = src_pts[mask == 1]
dst_ptsN = dst_pts[mask == 1]
print(len(src_ptsN))

e1 = findEpipole(F, src_ptsN)
e2 = findEpipole(F.T, dst_ptsN)

lineimg1 = img1
lineimg2 = img2

samplsrc = random.sample(list(src_ptsN), 10)
sampldst = random.sample(list(dst_ptsN), 10)

for i, j in zip(samplsrc, samplsrc):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    cv2.line(lineimg1, (int(i[0]), int(i[1])), (int(e2[0]), int(e2[1])), (r, g, b), 5)
    cv2.line(lineimg2, (int(j[0]), int(j[1])), (int(e1[0]), int(e1[1])), (r, g, b), 5)

cv2.imwrite("1epipolarLine.jpg", lineimg1)
cv2.imwrite("2epipolarLine.jpg", lineimg2)

print(e1)
print(e2)
print(F)
