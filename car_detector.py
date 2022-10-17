import cv2
import math
import numpy as np


def detect_cars(img):

#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.max(img, 0).astype('uint8')
    ret, blobs = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    #cv2.imshow('blobs', blobs)

    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))

    contours_sorted_filtered = []

    contours,hierarchy = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #biggest = None
    #biggestArea = 0
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 10000: # TODO mo
            contours_sorted_filtered.append(cnt)
    #        biggest = cnt
    #        biggestArea = M['m00']
    #    rect = cv2.minAreaRect(cnt)
    #    box = cv2.boxPoints(rect)
    #    box = np.int0(box)

    #blobs = cv2.cvtColor(blobs, cv2.COLOR_GRAY2BGR)

    #cnt = None
    #if biggest is not None:
    #    cnt = biggest
    #    rect = cv2.minAreaRect(cnt)
    #    box = cv2.boxPoints(rect)
    #    box = np.int0(box)
    #    cv2.drawContours(blobs,[box],0,(0,0,255),2)
    #    cv2.drawContours(img,[box],0,(0,0,255),2)

    #cv2.imshow('blobs_cl', blobs)
    return contours_sorted_filtered

if __name__ == '__main__':
    img = cv2.imread('pic.png')
    detect_car(img)
    cv2.waitKey()
