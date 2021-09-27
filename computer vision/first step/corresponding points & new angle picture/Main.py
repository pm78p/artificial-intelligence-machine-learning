import cv2
import numpy as np
import math
import random
from operator import itemgetter
from matplotlib import pyplot as plt
from matplotlib import colors


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


# this implement as told in the powerpoints
def RANSAC(matches, kp1, kp2, thr, p=0, sampleNumber=0):
    if sampleNumber <= 0:
        sampleNumber = int(len(matches) / 4)
    else:
        sampleNumber = int(sampleNumber)

    if p <= 0:
        p = 0.9
    counterInRANSAC = 0
    wMinInRANSAC = 0
    NinRANSAC = 10000  # like 10000 is infinity
    lhml = []  # List of Homography Matrices with their Inliers
    boolean = True
    while boolean:
        sample = random.sample(matches, 4)

        src_pts = np.float64([kp1[m.queryIdx].pt for m in sample]).reshape(-1, 1, 2)
        dst_pts = np.float64([kp2[m.trainIdx].pt for m in sample]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts)

        homoMatric = H
        inliers, inlierslist, outliers = calculateInliers(matches, kp1, kp2, homoMatric, thr, sampleNumber)
        lhml.append((inliers, homoMatric, inlierslist, outliers))
        wTmp = inliers / sampleNumber

        if wTmp > wMinInRANSAC:
            wMinInRANSAC = wTmp
            if wTmp == 1:
                break
            NinRANSAC = int(math.log(1 - p, 10) / math.log(1 - wTmp ** 4, 10)) + 1

        if counterInRANSAC >= NinRANSAC:
            boolean = False

        counterInRANSAC += 1

    bestHomoMatric = max(lhml, key=itemgetter(0))[1]
    inli = max(lhml, key=itemgetter(0))[2]
    outliers = max(lhml, key=itemgetter(0))[3]
    src_pts = np.float64([kp1[m.queryIdx].pt for m in inli]).reshape(-1, 1, 2)
    dst_pts = np.float64([kp2[m.trainIdx].pt for m in inli]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts)
    print(max(lhml, key=itemgetter(0))[0])
    print(counterInRANSAC)
    return H, inli, outliers


# its just check how many inliers are exist with temporary homographic matrices
def calculateInliers(matches, kp1, kp2, guesMatris, thrDist, sampleNumber=0):
    inlierslist = []
    outliers = []
    inliersCounter = 0
    sample = random.sample(matches, sampleNumber)
    for i in sample:
        img1_idx = i.queryIdx
        img2_idx = i.trainIdx
        x1, y1 = kp1[img1_idx].pt
        x2, y2 = kp2[img2_idx].pt

        x2proj, y2proj, z = np.dot(guesMatris, np.array([x1, y1, 1]))
        if z == 0:
            continue
        x2proj /= z
        y2proj /= z
        dist = math.sqrt((x2proj - x2) ** 2 + (y2proj - y2) ** 2)

        if dist <= thrDist:
            inlierslist.append(i)
            inliersCounter += 1
        else:
            outliers.append(i)

    return inliersCounter, inlierslist, outliers


# it derive the corresponds with implemented sift in opencv
def findCoresponds(img1, img1g, img2, img2g, sift):
    sift2 = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift2.detectAndCompute(img1g, None)
    kp2, des2 = sift2.detectAndCompute(img2g, None)

    match = cv2.BFMatcher()
    matches = match.knnMatch(des2, des1, k=2)

    return (kp2, kp1, matches)


# reading images
img1 = cv2.imread("11n.jpg")
img1g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("12n.jpg")
img2g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# make and sift object and derive keypoints and descriptors for both picturse
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1g, None)
kp2, des2 = sift.detectAndCompute(img2g, None)

# draw keypoints in each picture
keypoints1 = cv2.drawKeypoints(img1, kp1, None, color=(0, 130, 0))
keypoints2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 130, 0))

cv2.imwrite("drawkeypoints1.jpg", keypoints1)
cv2.imwrite("drawkeypoints2.jpg", keypoints2)

# some parameters for matrices they use for stiching the picturs
tool = np.shape(img1)[0]  # + np.shape(img2)[0]
arz = np.shape(img1)[1] + np.shape(img2)[1]

# the arrays below use for stcihing the pictures
dst = np.zeros((tool, arz, 3), np.float64)
dst[0:np.shape(img1)[0], 0:np.shape(img1)[1]] += keypoints1[0:np.shape(keypoints1)[0], 0:np.shape(keypoints1)[1]]

dst[0:np.shape(img2)[0], np.shape(img1)[1]: np.shape(img1)[1] + np.shape(img2)[1]] += keypoints2[
                                                                                      0:np.shape(keypoints2)[0],
                                                                                      0:np.shape(keypoints2)[1]]

cv2.imwrite("r14_corners.jpg", dst)

# use a matcher to match the keypoints to find coressponding points
match = cv2.BFMatcher()
matches = match.knnMatch(des1, des2, k=2)

# filtering the corresponding points with their distances parameter
matches1to2 = []
for i, j in matches:
    if i.distance / j.distance < 8 / 10:
        matches1to2.append(i)

# these two parameters use to painting the picturs with their corresponds and keypoints
src_pts = []
dst_pts = []
for m in matches1to2:
    src_pts.append(kp1[m.queryIdx])
for m in matches1to2:
    dst_pts.append(kp2[m.trainIdx])

# here we load the last picture wich are painted to repaint the new points or corresponds
drawkeypoints1 = cv2.imread("drawkeypoints1.jpg")
drawkeypoints2 = cv2.imread("drawkeypoints2.jpg")
keypoints1_2 = cv2.drawKeypoints(drawkeypoints1, src_pts, None, color=(180, 0, 0))
keypoints2_2 = cv2.drawKeypoints(drawkeypoints2, dst_pts, None, color=(180, 0, 0))
cv2.imwrite("drawkeypoints1.jpg", keypoints1_2)
cv2.imwrite("drawkeypoints2.jpg", keypoints2_2)
dst = np.zeros((tool, arz, 3), np.float64)
dst[0:np.shape(img1)[0], 0:np.shape(img1)[1]] += keypoints1_2[0:np.shape(keypoints1)[0], 0:np.shape(keypoints1)[1]]

dst[0:np.shape(img2)[0], np.shape(img1)[1]: np.shape(img1)[1] + np.shape(img2)[1]] += keypoints2_2[
                                                                                      0:np.shape(keypoints2)[0],
                                                                                      0:np.shape(keypoints2)[1]]

cv2.imwrite("r15_correspondences.jpg", dst)

# these part use for paint a line between the corresponding points
# first with dictionary match them
draw_params = dict(matchColor=(255, 0, 0), singlePointColor=None, flags=2)
# and with draw matches function we draw them
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches1to2, None, **draw_params)
cv2.imwrite("r16_SIFT.jpg", img3)

# here we do same but we use 20 random point to doing what we done in previous codes
draw_params = dict(matchColor=(255, 0, 0), singlePointColor=None, flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, random.sample(matches1to2, 20), None, **draw_params)
cv2.imwrite("r17.jpg", img3)

# here we filter the points again the parameter = 0.58 and i expriment it
matches1to2 = []
for i, j in matches:
    if i.distance / j.distance < 5.8 / 10:
        matches1to2.append(i)

# here we derive the homography matrix and inliers points with ransac and other functions
H, inliers, outliers = RANSAC(matches1to2, kp1, kp2, 10, 0.6, len(matches1to2))

# warping
dst2 = cv2.warpPerspective(img1, H, (arz, arz))
cv2.imwrite("r20.jpg", autocrop(dst2))

# again painting the points with same approach these are inliers
src_pts = []
dst_pts = []
for m in inliers:
    src_pts.append(kp1[m.queryIdx])
for m in inliers:
    dst_pts.append(kp2[m.trainIdx])

drawkeypoints1 = cv2.imread("drawkeypoints1.jpg")
drawkeypoints2 = cv2.imread("drawkeypoints2.jpg")
keypoints1_2 = cv2.drawKeypoints(drawkeypoints1, src_pts, None, color=(0, 0, 255))
keypoints2_2 = cv2.drawKeypoints(drawkeypoints2, dst_pts, None, color=(0, 0, 255))
dst = np.zeros((tool, arz, 3), np.float64)
dst[0:np.shape(img1)[0], 0:np.shape(img1)[1]] += keypoints1_2[0:np.shape(keypoints1)[0], 0:np.shape(keypoints1)[1]]

dst[0:np.shape(img2)[0], np.shape(img1)[1]: np.shape(img1)[1] + np.shape(img2)[1]] += keypoints2_2[
                                                                                      0:np.shape(keypoints2)[0],
                                                                                      0:np.shape(keypoints2)[1]]

cv2.imwrite("r18.jpg", dst)

for m in outliers:
    src_pts.append(kp1[m.queryIdx])
for m in outliers:
    dst_pts.append(kp2[m.trainIdx])

keypoints1_2 = cv2.drawKeypoints(img1, src_pts, None, color=(0, 255, 0))
keypoints2_2 = cv2.drawKeypoints(img2, dst_pts, None, color=(0, 255, 0))
dst = np.zeros((tool, arz, 3), np.float64)
dst[0:np.shape(img1)[0], 0:np.shape(img1)[1]] += keypoints1_2[0:np.shape(keypoints1)[0], 0:np.shape(keypoints1)[1]]

dst[0:np.shape(img2)[0], np.shape(img1)[1]: np.shape(img1)[1] + np.shape(img2)[1]] += keypoints2_2[
                                                                                      0:np.shape(keypoints2)[0],
                                                                                      0:np.shape(keypoints2)[1]]

cv2.imwrite("r19_mistakes.jpg", dst)

print(H)
