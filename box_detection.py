import cv2
import numpy as np


def detect_box(frame):
    """ Box detection of the drift box

    Parameters
    ----------
    frame : numpy.ndarray
            One image frame

    Returns
    -------
    x : int
        x cordinate of a corner of the rectangle
    y : int
        y cordinate of a corner of the rectangle
    w : int
        width of the rectangle
    h : int
        height of the rectangle
    theta : float
        degree of rotation
    box : numpy.ndarray
        cordinates of the corners
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blur, 255, 255)
    contours, hier = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    maxa = 0 #final max area of contour
    for c in contours: # find largest area of contours
        box = cv2.minAreaRect(c)
        h = int(box[1][0])
        w = int(box[1][1])
        a = w*h
        if maxa < a:
            maxa = a
            biggest = c

    box = cv2.minAreaRect(biggest)
    h = int(box[1][0])
    w = int(box[1][1])
    theta = box[2]
    box = cv2.boxPoints(box)
    box = np.int0(box)
    x = box[0][0]
    y = box[0][1]
    return x, y, w, h, theta, box