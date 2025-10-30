from typing import List
from itertools import combinations

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        def is_valid_rectangle(p1, p2, p3, p4):
            # Check if the points form a valid rectangle with edges parallel to the axes
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4

            # Check if opposite sides are parallel to the axes
            if x1 == x2 and y1 == y3 and x3 == x4 and y2 == y4:
                # Check if there are no other points inside or on the border of the rectangle
                for p in points:
                    x, y = p
                    if x1 < x < x3 and y1 < y < y2:
                        return False
                return True
            return False

        max_area = -1
        for p1, p2, p3, p4 in combinations(points, 4):
            if is_valid_rectangle(p1, p2, p3, p4):
                x1, y1 = p1
                x3, y3 = p3
                area = abs(x3 - x1) * abs(y3 - y1)
                max_area = max(max_area, area)

        return max_area