import cv2
import numpy as np

THRESHOLD = 20
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None


def draw_hist_rectangles(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, kernel, dst)

    hist_threshold_min = 100
    hist_threshold_max = 255
    _, thresh = cv2.threshold(dst, hist_threshold_min, hist_threshold_max, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))

    hist_mask_image = cv2.dilate(thresh, None, iterations=2)
    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)

    return cv2.bitwise_and(frame, hist_mask_image), thresh


def run_avg(image, background, alpha):
    if background is None:
        return image.copy().astype("float")

    # accumulate the weighted average background
    cv2.accumulateWeighted(image, background, alpha)
    return background


def find_hand_contour(image, background):
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # dilated_image = cv2.dilate(thresh, None, iterations=2)
    # eroded_image = cv2.erode(dilated_image, None, iterations=2)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)

    # get the contours in the thresholded image
    contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Old one:
    # cont, _ = cv2.findContours(hand_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(contours) == 0:
        return None, None
    else:
        # based on contour area, get the maximum contour which is the hand
        return max(contours, key=cv2.contourArea), closing
