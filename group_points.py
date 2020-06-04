"""This module is used to group the hull points around a user's fingertips."""

from math import sqrt
from dataclasses import dataclass

GROUP_DISTANCE = 40  # Max euclidean distance
MAX_X_DISTANCE = 20  # Max horizontal distance
MAX_Y_DISTANCE = 50  # Max vertical distance


@dataclass
class FlaggedPoint:
    """Data class that holds a point and a flag. Useful when iterating over points to
    keep track of which points need to be disincluded from further calculations."""
    point: tuple
    flag: bool


def group_fingertip_points(points):
    """Used for grouping hull points around fingertips. Hull points near the fingertips are first grouped.
    Then, if there are multiple hull points along the finger when the finger is near-upright, these points
    are grouped too."""
    # Flag here indicates the point has been grouped.
    points = [FlaggedPoint(point=point, flag=False) for point in points]
    fused_points = []
    n = len(points)
    for i in range(n):
        fi = points[i]
        if not fi.flag:
            fi.flag = True
            count = 1
            point = list(fi.point)
            for j in range(i + 1, n):
                fj = points[j]
                if euclidean_distance(fi.point, fj.point) < GROUP_DISTANCE:
                    point[0] += fj.point[0]
                    point[1] += fj.point[1]
                    count += 1
                    fj.flag = True
            point[0] /= count
            point[1] /= count
            fused_points.append(tuple(point))

    # Flag here indicates the point should be returned by the function.
    fused_points = [FlaggedPoint(point=point, flag=True) for point in fused_points]

    for i in range(len(fused_points) - 1):
        for j in range(i + 1, len(fused_points)):
            p1, p2 = fused_points[i].point, fused_points[j].point
            if (horizontal_distance(p1, p2) < MAX_X_DISTANCE and
                    vertical_distance(p1, p2) < MAX_Y_DISTANCE and
                    fused_points[i].flag is True and
                    fused_points[j].flag is True):
                if p1[1] > p2[1]:
                    fused_points[i].flag = False
                else:
                    fused_points[j].flag = False

    return [p.point for p in fused_points if p.flag is True]

def horizontal_distance(p1, p2):
    return abs(p1[0] - p2[0])

def vertical_distance(p1, p2):
    return abs(p1[1] - p2[1])

def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
