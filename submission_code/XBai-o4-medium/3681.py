import itertools
from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        max_area = -1
        n = len(points)
        if n < 4:
            return -1
        for combo in itertools.combinations(points, 4):
            # Extract x and y coordinates
            x_coords = [p[0] for p in combo]
            y_coords = [p[1] for p in combo]
            unique_x = set(x_coords)
            unique_y = set(y_coords)
            if len(unique_x) == 2 and len(unique_y) == 2:
                x1, x2 = sorted(unique_x)
                y1, y2 = sorted(unique_y)
                # Create corner set
                corner_set = set()
                for p in combo:
                    corner_set.add((p[0], p[1]))
                # Check all points
                invalid = False
                for p in points:
                    px, py = p[0], p[1]
                    if (px, py) in corner_set:
                        continue
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        invalid = True
                        break
                if not invalid:
                    current_area = (x2 - x1) * (y2 - y1)
                    if current_area > max_area:
                        max_area = current_area
        return max_area if max_area != -1 else -1