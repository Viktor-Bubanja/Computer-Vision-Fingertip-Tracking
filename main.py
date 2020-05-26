import time

import cv2

from fingertip_tracking import find_fingertip
from hand_segmentation import find_hand_contour, run_avg, draw_hist_rectangles, hand_histogram

convex_defects = []

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main():
    background = None
    # initialize alpha weight for running average
    alpha = 0.5
    cap = cv2.VideoCapture(0)
    num_frames = 100
    frame_count = 0
    frame = None

    while cv2.waitKey(1) & 0xFF != ord('z'):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = draw_hist_rectangles(frame)
        cv2.imshow("Retrieve histogram", rescale_frame(frame))

    hand_hist = hand_histogram(frame)

    time.sleep(2)

    while cv2.waitKey(1) & 0xFF != ord('q'):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale and apply Guassian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if frame_count < num_frames:
            # Note: background will be None for the first iteration.
            background = run_avg(gray, background, alpha)
            frame_count += 1
            cv2.imshow("Calibrating background", frame)

        else:
            hand_region, thresh = find_hand_contour(gray, background)
            if thresh is not None:
                cv2.imshow("Thresh", rescale_frame(thresh))

            fingertips = find_fingertip(frame, hand_region)
            fingertips.sort(key=lambda x: x[0])
            for finger in fingertips:
                cv2.circle(frame, finger, 5, [255, 255, 255], -1)

            cv2.imshow("Live Feed", rescale_frame(frame))


    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
