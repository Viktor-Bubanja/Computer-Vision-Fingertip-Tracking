import cv2
import numpy as np

hand_hist = None
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

finger_path = []


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


"""
Draw rectangles on the frame to indicate to the user where they should place their hand.
The colours of the pixels within the rectangles are later extracted to generate a histogram.
"""
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


"""
Extract pixels within rectangles and generate HSV histogram.
"""
def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


"""
Given a frame and a histogram, finds all regions that match the histogram using Histogram Back Projection.
Returns a frame containing only these features.
"""
def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    # threshold the image, then perform erosion and dilation to remove any small regions of noise
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))

    hist_mask_image = cv2.erode(thresh, None, iterations=2)
    hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)
    return cv2.bitwise_and(frame, hist_mask_image)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])
    return cx, cy


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        dist = cv2.subtract(cy, y)
        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            return tuple(contour[farthest_defect][0])
        else:
            return 0, 0


def draw_circles(frame, point_path):
    last_point = point_path[-1]
    for i in range(len(point_path) - 1):
        cv2.circle(frame, point_path[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)
    cv2.circle(frame, last_point, 5, [0, 0, 255], -1)


"""
Identify a pointed finger in a frame by finding a convexity defect furthest from the centroid of the contour.
"""
def find_fingertip(frame, hist_mask_image):
    global finger_path
    contour_list = contours(hist_mask_image)
    if contour_list:
        max_cont = max(contour_list, key=cv2.contourArea)
        cnt_centroid = centroid(max_cont)
        drawable_hull = cv2.convexHull(max_cont)
        cv2.drawContours(frame, [drawable_hull], -1, (255, 0, 0), 2)
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)

        far_point = farthest_point(defects, max_cont, cnt_centroid)

        print(far_point)
        return far_point
    else:
        return None



def main():
    global hand_hist
    cap = cv2.VideoCapture(0)

    frame = None

    while cv2.waitKey(1) & 0xFF != ord('z'):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        draw_hist_rectangles(frame)
        cv2.imshow("Live Feed", rescale_frame(frame))

    hand_hist = hand_histogram(frame)

    while cv2.waitKey(1) & 0xFF != ord('q'):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        hist_mask_image = hist_masking(frame, hand_hist)

        far_point = find_fingertip(frame, hist_mask_image)
        if far_point:
            finger_path.append(far_point)
            if len(finger_path) > 20:
                finger_path.pop(0)

            draw_circles(frame, finger_path)
        cv2.imshow("Live Feed", rescale_frame(frame))

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
