from typing import List
import itertools

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        if len(points) < 4:
            return -1
        points_set = set((x, y) for x, y in points)
        unique_x = sorted({x for x, y in points})
        unique_y = sorted({y for x, y in points})
        if len(unique_x) < 2 or len(unique_y) < 2:
            return -1
        max_area = -1
        for x1, x2 in itertools.combinations(unique_x, 2):
            for y1, y2 in itertools.combinations(unique_y, 2):
                corners = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}
                if corners.issubset(points_set):
                    valid = True
                    for p in points:
                        px, py = p[0], p[1]
                        if (px, py) not in corners:
                            if x1 <= px <= x2 and y1 <= py <= y2:
                                valid = False
                                break
                    if valid:
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
        return max_area if max_area != -1 else -1