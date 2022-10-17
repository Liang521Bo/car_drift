import cv2
import numpy as np
import car_detector
from overlap_analyzer import OverlapAnalyzer
from car_analyzer import CarAnalyzer
from box_detection import detect_box
from car_detection_bg_sub import CarDetectorBackgroundSub
from effects import Effects

#FILENAME="videos/Video_liang/drifting.mp4"
FILENAME="example_input.mp4"
WAITKEY_DELAY=1
#FILENAME=0
vid_reader = cv2.VideoCapture(FILENAME)

#vid_writer =cv2.VideoWriter("example_run.mp4", cv2.VideoWriter.fourcc(*'mp4v'), 30, (1920, 1080))

for i in range(30):
    # skip some initial frames as things are likely to be moving, camera autofocusing etc
    success, frame = vid_reader.read()

#TODO: smarter background detection? update per-frame?
background = frame.astype('float32')

car_fg_detector = CarDetectorBackgroundSub()
x,y,w,h,theta,driftBox = detect_box(frame)
car_analyzers = []
#overlap_analyzers = []
#overlap_analyzer = OverlapAnalyzer(frame, driftBox)
#car_analyzer = CarAnalyzer()
#effects = Effects()

paused = False
step = False

# how long an analyzer should look for a car before giving up
old_analyzer_limit = 10
maximum_match_distance = 500



while success:
    if not paused or step:
        success, frame_orig = vid_reader.read()
        if not success:
            break
        #background = background * 0.95 + frame_orig.astype('float32') * 0.05
        #diff = cv2.absdiff(frame.astype('float32'), background.astype('float32'))
        frame = frame_orig.copy()
        fg = car_fg_detector.get_fg(frame)
        cnts = car_detector.detect_cars(fg)
        if len(cnts) > 0:
            cnt = cnts[0]
        else:
            cnt = None

        frame = cv2.drawContours(frame,[driftBox],0,(0,0,255),2)
#        frame = cv2.drawContours(frame,cnts,-1,(0,0,255),2)


        for cnt in cnts:
            closest = None
            closest_dist = 0
            for analyzer in car_analyzers:
                if analyzer.frames_since_seen > 0:
                    rect = cv2.minAreaRect(cnt)
                    distance = np.sqrt((rect[0][0] - analyzer.last_seen_pos[0]) ** 2 + (rect[0][1] - analyzer.last_seen_pos[1]) ** 2)
                    if distance < maximum_match_distance and (distance < closest_dist or closest is None):
                        closest_dist = distance
                        closest = analyzer
                        print("found closest: ", distance, analyzer.frames_since_seen)

            if closest is None:
                print("didn't find making new")
                closest = CarAnalyzer()
                closest.overlap_analyzer = OverlapAnalyzer(frame_orig, driftBox)
                car_analyzers.append(closest)

            frame = closest.update_car(cnt, frame)


        any_is_drifting = False
        for analyzer in car_analyzers:
            analyzer.update_time()
            frame = analyzer.overlap_analyzer.update(analyzer, frame)
            if analyzer.frames_since_seen > old_analyzer_limit:
                car_analyzers.remove(analyzer)
                print("bye bye")
            if analyzer.overlap_analyzer.is_drifting:
                any_is_drifting = True

#        if any_is_drifting:
#            frame = effects.apply(frame)


#        frame = overlap_analyzer.update(car_analyzer, frame)

        step = False
        #vid_writer.write(frame)
        cv2.imshow("frame", frame)
#        if car_analyzer.frames_since_seen < 15:
    #    cv2.imshow("diff", (diff / 255) * 2)

    k = cv2.waitKey(WAITKEY_DELAY)
#    else:
#        k = 0
    if k == ord('q'):
        break
    if k == ord(' '):
        paused = not paused
    if k == ord('n'):
        step = True
    if k == ord('s'):
        cv2.imwrite('screenshot.png', frame)
    if k == ord('u'):
        x,y,w,h,theta,driftBox = detect_box(frame_orig)
        overlap_analyzer = OverlapAnalyzer(frame_orig, driftBox)
