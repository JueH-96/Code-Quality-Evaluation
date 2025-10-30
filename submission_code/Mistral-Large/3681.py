from typing import List
from itertools import combinations

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        def is_valid_rectangle(p1, p2, p3, p4):
            # Check if the points form a rectangle with edges parallel to the axes
            if p1[0] != p2[0] or p3[0] != p4[0] or p1[1] != p3[1] or p2[1] != p4[1]:
                return False
            # Check if there are any points inside or on the border of the rectangle
            for px, py in points:
                if (px > p1[0] and px < p2[0] and py > p1[1] and py < p3[1]):
                    return False
            return True

        max_area = -1
        point_set = set(map(tuple, points))

        for p1, p2, p3, p4 in combinations(points, 4):
            if is_valid_rectangle(p1, p2, p3, p4):
                area = abs(p1[0] - p2[0]) * abs(p1[1] - p3[1])
                max_area = max(max_area, area)

        return max_area