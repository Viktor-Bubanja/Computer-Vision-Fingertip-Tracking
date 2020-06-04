"""Module responsible for finding the fingertips within a binary image."""

import cv2
import math
from group_points import group_fingertip_points


def find_fingertips(frame, hand_region):
    """Find fingertips by finding a convex hull of the user's hand, grouping hull points
    so there is one hull point per finger, find the number of significant convexity defects (X),
    then identify the topmost X + 1 hull points as fingertips."""
    fingertips = []
    if hand_region is not None:
        hull_points = cv2.convexHull(hand_region)
        cv2.drawContours(frame, [hull_points], -1, (255, 0, 0), 2)
        hull_points = [tuple(i[0]) for i in hull_points]
        hand_height = max(hull_points, key=lambda p: p[1])[1] - min(hull_points, key=lambda p: p[1])[1]
        defect_threshold = int(hand_height / 4)
        num_defects = find_number_convex_defects(hand_region, defect_threshold)
        fused_hull = sorted(group_fingertip_points(hull_points), key=lambda p: p[1])

        for i in range(num_defects + 1):
            if i < len(fused_hull):
                point = (int(fused_hull[i][0]), int(fused_hull[i][1]))
                fingertips.append(point)
    return fingertips

def find_number_convex_defects(hand_region, defect_threshold):
    """Find the number of significant convexity defects in a hand (corresponding to
    the gaps between fingers."""
    num_defects = 0
    hull = cv2.convexHull(hand_region, returnPoints=False)
    defects = cv2.convexityDefects(hand_region, hull)
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(hand_region[s][0])
            end = tuple(hand_region[e][0])
            far = tuple(hand_region[f][0])
            angle = find_angle_between_three_points(start, far, end)
            # check angle to avoid defects that are not between two fingers
            if angle <= math.pi / 2 and d > defect_threshold:
                num_defects += 1

    return min(num_defects, 4)

def find_angle_between_three_points(p1, p2, p3):
    a = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
    b = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    c = math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
    # Use cosine theorem to calculate angle between convex defect and surrounding hull points
    return math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
