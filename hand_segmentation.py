import cv2
import numpy as np


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

    thresh_min = 190
    thresh_max = 255
    _, thresh = cv2.threshold(dst, thresh_min, thresh_max, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))

    hist_mask_image = cv2.dilate(thresh, None, iterations=2)
    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)

    return hist_mask_image


def lowest_y_value(binary_image):
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour[max_contour[:, :, 1].argmax()][0][1]
    else:
        height, _ = binary_image.shape
        return height


def run_avg(image, background, alpha):
    if background is None:
        return image.copy().astype("float")

    # accumulate the weighted average background
    cv2.accumulateWeighted(image, background, alpha)
    return background

def grayscale_blur_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (7, 7), 0)


def background_substraction(image, background, lowest_point):
    gray_image = grayscale_blur_image(image)

    # find the absolute difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), gray_image)

    # threshold the diff image so that we get the foreground
    threshold = 20
    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    return thresh

def remove_points_below(image, cutoff):
    height, width = image.shape
    for i in range(cutoff, height):  # highlight pixels with any R or G or B values
        for j in range(0, width):
            image[i, j] = 0
    return image

def find_hand_contour(image, hand_hist, background):
    hist_thresh = hist_masking(image, hand_hist)
    lowest_point = lowest_y_value(hist_thresh)
    background_thresh = background_substraction(image, background, lowest_point)
    thresh = remove_points_below(background_thresh, lowest_point)
    thresh = cv2.merge((thresh, thresh, thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    cv2.imshow("background", background_thresh)
    cv2.imshow("hist", hist_thresh)

    # get the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(contours) == 0:
        return None
    else:
        # based on contour area, get the maximum contour which is the hand
        return max(contours, key=cv2.contourArea)
