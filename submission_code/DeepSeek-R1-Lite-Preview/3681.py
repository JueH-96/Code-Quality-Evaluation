from typing import List
from itertools import combinations

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        if len(points) < 4:
            return -1
        
        points_set = set((x, y) for x, y in points)
        
        x_to_ys = {}
        for x, y in points:
            if x in x_to_ys:
                x_to_ys[x].add(y)
            else:
                x_to_ys[x] = set([y])
        
        xs = sorted(x_to_ys.keys())
        max_area = -1
        
        for x1, x2 in combinations(xs, 2):
            common_ys = sorted(x_to_ys[x1].intersection(x_to_ys[x2]))
            if len(common_ys) >= 2:
                for y1, y2 in combinations(common_ys, 2):
                    # Check if all four corners are present
                    if (x1, y1) in points_set and (x1, y2) in points_set and (x2, y1) in points_set and (x2, y2) in points_set:
                        # Check no other point lies inside or on the border
                        valid = True
                        for point in points:
                            px, py = point
                            if x1 <= px <= x2 and y1 <= py <= y2 and (px, py) not in {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}:
                                valid = False
                                break
                        if valid:
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
        return max_area