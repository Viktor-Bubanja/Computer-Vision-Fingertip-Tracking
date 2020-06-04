import time

import cv2

from fingertip_tracking import find_fingertip
from hand_segmentation import find_hand_contour, run_avg, draw_hist_rectangles, hand_histogram

convex_defects = []

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def grayscale_blur_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (7, 7), 0)


def main2():
    cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) & 0xFF != ord('z'):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = draw_hist_rectangles(frame)
        cv2.imshow("Retrieve histogram", rescale_frame(frame))

    hand_hist = hand_histogram(frame)

    while cv2.waitKey(1) & 0xFF != ord('q'):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

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

    cv2.destroyWindow("Retrieve histogram")

    while cv2.waitKey(1) & 0xFF != ord('q'):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if frame_count < num_frames:
            # Note: background will be None for the first iteration.
            gray_background = grayscale_blur_image(frame)
            background = run_avg(gray_background, background, alpha)
            frame_count += 1
            cv2.imshow("Calibrating background", frame)

        else:
            cv2.destroyWindow("Calibrating background")
            hand_region = find_hand_contour(frame, hand_hist, background)

            fingertips = find_fingertip(frame, hand_region)
            fingertips.sort(key=lambda x: x[0])
            for finger in fingertips:
                cv2.circle(frame, finger, 5, [255, 255, 255], -1)

            cv2.imshow("Live Feed", rescale_frame(frame))


    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
