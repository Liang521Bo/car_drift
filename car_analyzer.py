import cv2
import numpy as np

class CarAnalyzer:
    def __init__(self):
        self.prevCarBox = None
        self.prevCarRect = None
        self.carBox = None
        self.carRect = None
        self.angle = 0
        self.dirAngle = 0
        self.frames_since_seen = 0
        self.last_seen_pos = [0,0]
        self.overlap_analyzer = None

    def update_time(self):
        self.frames_since_seen += 1

    def update_car(self, contour, frame):
        self.prevCarRect = self.carRect
        self.prevCarBox = self.carBox
        cv2.putText(frame, f"frames since track={self.frames_since_seen}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

        if contour is not None:
            carRect = cv2.minAreaRect(contour)
            center, (width, height), angle = carRect
            if width < 100 or height < 100:
                return frame
            # this is ugly, as it assumes the box is on the bottom of the screen with the car going to the right
            if width < height:
                height, width = width, height
                angle = angle - 90
            self.carRect = (center, (width, height), angle)
            self.carBox = cv2.boxPoints(self.carRect)
            self.carBox = np.int0(self.carBox)
            cv2.drawContours(frame,[self.carBox],0,(0,0,255),2)
            self.frames_since_seen = 0
            self.last_seen_pos = center

            if self.prevCarRect is not None:
                self.lastFound = True
                pointDiff = np.asarray(self.carRect[0]) - np.asarray(self.prevCarRect[0])
                angle = self.carRect[2]
                dirAngle = np.rad2deg(np.arctan2(pointDiff[1], pointDiff[0]))
                self.angle = angle
                self.dirAngle = dirAngle
                dist = np.linalg.norm(pointDiff)
                cv2.putText(frame, f"angle={angle - dirAngle}", self.carBox[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                framerate = 30
                cv2.putText(frame, f"vel={dist/(1/framerate)}", np.int0(np.asarray(self.prevCarRect[0]) + np.asarray([100, 0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                cv2.line(frame, np.int0(self.carRect[0]), np.int0(self.carRect[0] + pointDiff), (0, 255, 0), 3)
                cv2.line(frame, np.int0(self.carRect[0]), np.int0(self.carRect[0] + np.asarray([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) * 50), (255, 0, 0), 3)

                cv2.putText(frame, f"angle={angle - dirAngle}", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                framerate = 30
                cv2.putText(frame, f"vel={dist/(1/framerate)}", np.int0(np.asarray(self.prevCarRect[0]) + np.asarray([100, 0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        return frame
