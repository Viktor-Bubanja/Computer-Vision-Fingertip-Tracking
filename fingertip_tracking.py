import cv2
import math

from dataclasses import dataclass

DEFECT_THRESHOLD = 10000
THUMB_THRESHOLD = 100
MIN_CONTOUR_AREA = 20000
GROUPING_Y = 30
GROUPING_X = 30


def find_fingertip(frame, hand_region):
    fingertips = []
    convex_defects = []
    if hand_region is not None:
        drawable_hull = cv2.convexHull(hand_region)
        cv2.drawContours(frame, [drawable_hull], -1, (255, 0, 0), 2)
        formatted = [tuple(i[0]) for i in drawable_hull]
        # Sort convex hull points by their y-value.
        print(formatted)
        group_distance = 40  # hand_width / 5
        fused_hull = sorted(fuse(frame, formatted, group_distance), key=lambda point: point[1])

        hull = cv2.convexHull(hand_region, returnPoints=False)
        defects = cv2.convexityDefects(hand_region, hull)
        if defects is not None:
            hand_height = max(formatted, key=lambda x: x[1])[1] - min(formatted, key=lambda x: x[1])[1]
            defect_threshold = int(hand_height / 4)
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                start = tuple(hand_region[s][0])
                end = tuple(hand_region[e][0])
                far = tuple(hand_region[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                # Use cosine theorem to calculate angle between convex defect and surrounding hull points
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if angle <= math.pi / 2 and d > defect_threshold:
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


def horizontal_distance(p1, p2):
    return abs(p1[0] - p2[0])


def vertical_distance(p1, p2):
    return abs(p1[1] - p2[1])


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def fuse(frame, points, max_distance):
    fused_points = []
    length = len(points)
    taken = [False] * length
    for i in range(length):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i + 1, length):
                p1, p2 = points[i], points[j]
                if euclidean_distance(p1, p2) < max_distance:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            flagged_point = FlaggedPoint(point=(int(point[0]), int(point[1])), flag=True)
            fused_points.append(flagged_point)


    # for point in fused_points:
    #     cv2.circle(frame, point.point, 10, [255, 255, 255], -1)

    max_x_distance = 20
    max_y_distance = 30

    for i in range(len(fused_points) - 1):
        for j in range(i + 1, len(fused_points)):
            p1, p2 = fused_points[i].point, fused_points[j].point
            if (horizontal_distance(p1, p2) < max_x_distance and
                vertical_distance(p1, p2) < max_y_distance and
                    fused_points[i].flag is True and
                    fused_points[j].flag is True):
                if p1[1] > p2[1]:
                    fused_points[i].flag = False
                else:
                    fused_points[j].flag = False

    # for point in fused_points:
    #     if point.flag is True:
    #         cv2.circle(frame, point.point, 5, [10, 30, 30], -1)

    return [point.point for point in fused_points if point.flag is True]


@dataclass
class FlaggedPoint:
    point: tuple
    flag: bool

