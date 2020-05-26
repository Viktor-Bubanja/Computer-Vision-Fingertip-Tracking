import cv2
import math

from hand_segmentation import find_hand_contour, run_avg

DEFECT_THRESHOLD = 12000
THUMB_THRESHOLD = 100
MIN_CONTOUR_AREA = 20000
GROUPING_Y = 30
GROUPING_X = 30

finger_path = []


def find_centroid(contour):
    moment = cv2.moments(contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return 0, 0


def contours(hand_region):
    # _, thresh = cv2.threshold(hand_region, 0, 255, cv2.THRESH_BINARY)
    cont, _ = cv2.findContours(hand_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def find_fingertip(frame, hand_region):
    fingertips = []
    convex_defects = []
    if hand_region is not None:
        centroid = find_centroid(hand_region)
        cv2.circle(frame, centroid, 3, [20, 160, 50], -1)
        drawable_hull = cv2.convexHull(hand_region)
        cv2.drawContours(frame, [drawable_hull], -1, (255, 0, 0), 2)
        formatted = [tuple(i[0]) for i in drawable_hull]
        # Sort convex hull points by their y-value.
        hand_width = max(formatted, key=lambda x: x[0])[0] - min(formatted, key=lambda x: x[0])[0]

        group_distance = 30 # hand_width / 5
        fused_hull = sorted(fuse(formatted, group_distance), key=lambda point: point[1])

        hull = cv2.convexHull(hand_region, returnPoints=False)
        defects = cv2.convexityDefects(hand_region, hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                _, _, f, d = defects[i, 0]
                far = tuple(hand_region[f][0])
                if d > DEFECT_THRESHOLD and far[1] < centroid[1] + THUMB_THRESHOLD:
                    convex_defects.append(far)


        convex_defects.sort(key=lambda defect: defect[1])
        convex_defects = convex_defects[:5]
        for defect in convex_defects:
            cv2.circle(frame, defect, 5, [0, 255, 0], -1)

        for i in range(len(convex_defects) + 1):
            if i < len(fused_hull):
                point = (int(fused_hull[i][0]), int(fused_hull[i][1]))
                fingertips.append(point)

    return fingertips


def horizontalDistance(p1, p2):
    return p1[0] - p2[0]


def verticalDistance(p1, p2):
    return p1[1] - p2[1]

def dist2(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def fuse(points, distance):
    print(distance)
    ret = []
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                if dist2(p1, p2) < distance:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret





