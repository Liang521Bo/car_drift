import cv2
import random

class Effects:
    def __init__(self):
        self.speedlines = []
        speedlines_reader = cv2.VideoCapture("./[Effect] Speedlines Overlay (MP4 and PCF download)-t7cs11HaqEA.mp4")
        succ, frame = speedlines_reader.read()
        while succ:
            self.speedlines.append(frame)
            succ, frame = speedlines_reader.read()
        self.speedlines_idx = 0

    def apply(self, frame):
        shake_range = 10
        shift_x = random.randint(-shake_range+1,shake_range-1)
        shift_y = random.randint(-shake_range+1,shake_range-1)
        frame[shake_range:-shake_range, shake_range:-shake_range, :] = frame[shake_range+shift_x:-shake_range+shift_x, shake_range+shift_y:-shake_range+shift_y, :]
        frame = cv2.bitwise_or(frame, self.speedlines[self.speedlines_idx])
        self.speedlines_idx = (self.speedlines_idx + 1) % len(self.speedlines)
        return frame


