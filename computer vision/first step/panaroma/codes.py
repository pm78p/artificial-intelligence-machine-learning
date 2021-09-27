import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from operator import itemgetter
import math


# this is RANSAC function
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
        inliers, inlierslist = calculateInliers(matches, kp1, kp2, homoMatric, thr, sampleNumber)
        lhml.append((inliers, homoMatric, inlierslist))
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
    src_pts = np.float64([kp1[m.queryIdx].pt for m in inli]).reshape(-1, 1, 2)
    dst_pts = np.float64([kp2[m.trainIdx].pt for m in inli]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts)
    print(max(lhml, key=itemgetter(0))[0])
    return H


# its just check how many inliers are exist with temporary homographic matrices
def calculateInliers(matches, kp1, kp2, guesMatris, thrDist, sampleNumber=0):
    inlierslist = []
    inliersCounter = 0
    sample = random.sample(matches, sampleNumber)
    for i in sample:
        img1_idx = i.queryIdx
        img2_idx = i.trainIdx
        x1, y1 = kp1[img1_idx].pt
        x2, y2 = kp2[img2_idx].pt

        x2proj, y2proj, z = np.dot(guesMatris, np.array([x1, y1, 1]))
        x2proj /= z
        y2proj /= z
        dist = math.sqrt((x2proj - x2) ** 2 + (y2proj - y2) ** 2)

        if dist <= thrDist:
            inlierslist.append(i)
            inliersCounter += 1

    return inliersCounter, inlierslist


# the croper comes from internet and i just copy that because i think i could use cropping from paint toi hope i didnt make mistake
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


# it find the best middle picture and derive the corresponds with that by implemented sift in opencv
#this function has recursion to find middle picture
def findCoresponds(searchList, img2, img2g, imgInList):
    img1 = cv2.imread(searchList[0])
    img1g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(img1g, None)
    kp2, des2 = sift.detectAndCompute(img2g, None)

    match = cv2.BFMatcher()
    matches = match.knnMatch(des2, des1, k=2)

    someparameter = 2 / 10
    matches1to2 = []

    #in this loop we increse filter parametr to see wether the underprocces picture has the good correspond with current picture or not
    #every step we took in loop more corresponds appear but less reliabilty
    while someparameter < 5 / 10:
        if len(matches1to2) < 10:
            for i, j in matches:
                if i.distance / j.distance < someparameter:
                    matches1to2.append(i)
            someparameter += 1 / 10
        else:
            return (kp2, kp1, matches1to2)

    if searchList[1] == imgInList:
        return (kp2, kp1, matches1to2)
    searchList.remove(searchList[0])
    return findCoresponds(searchList, img2, img2g, imgInList)


sift = cv2.xfeatures2d.SIFT_create()

# list of pictures they listed because the loop add them to the orginal picture
listImg = ["07.jpg", "03.jpg", "04.jpg", "05.jpg", "10.jpg", "08.jpg", "09.jpg"]
# this is a dictunary that hold the homographic matrices based on center image wich is 06.jpg
# it means that every pictures has the own matric wich if you warp that picture with that the result that the picture u can stich it with center pic
# this dictionary use to find homo matrices for pictures wich are have not any correspond with center picture or have many corresponds with other picture in compair to center picture
dictH = {"06n.jpg": np.identity(3)}

# pana is the panaroma picture at the first its the center picture and here we read it
pana = cv2.imread("06n.jpg")
cv2.imwrite("r13.jpg", pana)

# in this loop we find each picture warp and add it to panaroma picture or pana (r13.jpg)

for imgInList in listImg:

    # this list is used for getting the middle picture that has corresponds and homo matrice with current point and center point
    list2 = ["06n.jpg", "07.jpg", "03.jpg", "04.jpg", "05.jpg", "10.jpg", "08.jpg", "09.jpg"]
    img2 = cv2.imread(imgInList)
    img2g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # panaroma picture size witch panaroma pictue and current picture
    tool = np.shape(pana)[0] + np.shape(img2)[0]  # np.shape(img2)[0] + np.shape(img1)[0]
    arz = np.shape(pana)[1] + np.shape(img2)[1] + 1000  # np.shape(img2)[1] + np.shape(img1)[1]

    #here we make and list to find middle picture
    #the latest pictures has the most priority and
    objectNumber = list2.index(imgInList)
    l = list2[0:objectNumber]
    l.reverse()
    list2 = l
    kp1, kp2, matches1to2 = findCoresponds(list2, img2, img2g, imgInList)
    img1 = cv2.imread(list2[0])

    #here we use corresponding points to derive our homographic matric
    src_pts = np.float64([kp1[m.queryIdx].pt for m in matches1to2]).reshape(-1, 1, 2)
    dst_pts = np.float64([kp2[m.trainIdx].pt for m in matches1to2]).reshape(-1, 1, 2)

    #with ransac func calculate the homographic matric
    H = RANSAC(matches1to2, kp1, kp2, 10, 0.99, len(matches1to2) * 4 / 4)
    #and here we dot the homographic matric with middle picture to find our orginal homographic matric in scale to (currennt --> center)
    H = np.dot(dictH[list2[0]], H)
    #and save the matric for other pictures
    dictH[imgInList] = H

    #here just warping current pic and stich it with panaroma
    dst = np.zeros((tool, arz, 3), np.float64)
    dst3 = np.zeros((tool, arz, 3), np.float64)
    pana = cv2.imread("r13.jpg")
    dst[0:np.shape(pana)[0], 0:np.shape(pana)[1]] += pana[0:np.shape(pana)[0], 0:np.shape(pana)[1]]
    dst2 = cv2.warpPerspective(img2, H, (arz, arz))
    dst3 += dst2[0:np.shape(dst3)[0], 0:np.shape(dst3)[1]]

    counter = 0
    for i, j in zip(dst, dst3):
        for k, t in zip(i, j):
            if k[0] == 0 and k[1] == 0 and k[2] == 0:
                k[0] = t[0]
                k[1] = t[1]
                k[2] = t[2]

    #and in every step we crop the result to be away of memory allocation
    dst = autocrop(dst)
    #and save the result in every steps to use it in next steps
    cv2.imwrite("r13.jpg", dst)
