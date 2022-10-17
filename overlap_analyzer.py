import cv2
import numpy as np

class OverlapAnalyzer():
    def __init__(self, frame_meta, driftBox):
        self.driftBox = driftBox # box for detecting overlap. type: cv2 contour
        self.frame_meta = frame_meta # used for size/data type
        self.overlap_accumulator = np.zeros(frame_meta.shape[:2], dtype=frame_meta.dtype)
        self.boxMask = cv2.drawContours(np.zeros(frame_meta.shape[:2], dtype=frame_meta.dtype), [driftBox], -1, 255, -1)
        self.last_carBox = None
        self.is_drifting = False


    def update(self, car_analyzer, frame):
        carBox = car_analyzer.carBox
        frames_since_seen = car_analyzer.frames_since_seen
        dirAngle = car_analyzer.dirAngle
        if carBox is not None:
            if self.last_carBox is not None:
                hull = cv2.convexHull(np.concatenate((carBox, self.last_carBox)))
                carMask = cv2.drawContours(np.zeros(self.frame_meta.shape[:2], dtype=self.frame_meta.dtype), [hull], -1, 255, -1)
            else:
                carMask = cv2.drawContours(np.zeros(self.frame_meta.shape[:2], dtype=self.frame_meta.dtype), [carBox], -1, 255, -1)
            overlap = cv2.bitwise_and(carMask, self.boxMask)
            overlap_count = cv2.countNonZero(overlap)
            self.is_drifting = overlap_count > 2000
            cv2.putText(frame, f"overlap={overlap_count}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            cv2.putText(frame, f"percent={100 * cv2.countNonZero(self.overlap_accumulator) / cv2.countNonZero(self.boxMask)}", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            # add to the accumulator
            self.overlap_accumulator = cv2.addWeighted(self.overlap_accumulator, 1, overlap, 1, 0)
            frame = cv2.addWeighted(frame, 1, cv2.cvtColor(self.overlap_accumulator, cv2.COLOR_GRAY2BGR), 0.3, 0)
            self.last_carBox = carBox
        return frame
