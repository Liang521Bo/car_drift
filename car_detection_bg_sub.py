import cv2
import numpy as np
import matplotlib.pyplot as plt



class CarDetectorBackgroundSub:
    def __init__(self):
        # create the background object, you can choose to detect shadows or not (if True they will be shown as gray)
        self.backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    def get_fg(self, frame):
        # apply the background object on each frame
        fgmask = self.backgroundObject.apply(frame)

        # Perform thresholding to get rid of the shadows.
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

        # also extracting the real detected foreground part of the image (optional)
        #real_part = cv2.bitwise_and(frame, frame, mask=fgmask)

        # making fgmask 3 channeled so it can be stacked with others
        #fgmask_3 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        # Stack all three frames and show the image
        #stacked = np.hstack((fgmask_3, frame, real_part))
        #cv2.imshow('All three', cv2.resize(stacked, None, fx=1, fy=1))

        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break
        return fgmask
