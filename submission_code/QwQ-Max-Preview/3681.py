from itertools import combinations
from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        if len(points) < 4:
            return -1
        
        points_set = set((x, y) for x, y in points)
        x_coords = sorted({x for x, y in points})
        y_coords = sorted({y for x, y in points})
        
        if len(x_coords) < 2 or len(y_coords) < 2:
            return -1
        
        max_area = -1
        
        for x_pair in combinations(x_coords, 2):
            x1, x2 = x_pair
            for y_pair in combinations(y_coords, 2):
                y1, y2 = y_pair
                corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
                if not all(corner in points_set for corner in corners):
                    continue
                
                valid = True
                for (px, py) in points:
                    if (px == x1 or px == x2) and (py == y1 or py == y2):
                        continue
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        valid = False
                        break
                
                if valid:
                    current_area = (x2 - x1) * (y2 - y1)
                    if current_area > max_area:
                        max_area = current_area
        
        return max_area if max_area != -1 else -1