from itertools import combinations
from collections import Counter
from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        n = len(points)
        if n < 4:
            return -1
        
        max_area = -1
        
        for combo in combinations(points, 4):
            xs = [p[0] for p in combo]
            ys = [p[1] for p in combo]
            
            x_counts = Counter(xs)
            y_counts = Counter(ys)
            
            if len(x_counts) != 2 or len(y_counts) != 2:
                continue
            
            valid_x = all(count == 2 for count in x_counts.values())
            valid_y = all(count == 2 for count in y_counts.values())
            
            if not (valid_x and valid_y):
                continue
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            area = (max_x - min_x) * (max_y - min_y)
            
            combo_set = {tuple(p) for p in combo}
            valid = True
            
            for p in points:
                if tuple(p) not in combo_set:
                    x, y = p
                    if min_x <= x <= max_x and min_y <= y <= max_y:
                        valid = False
                        break
            
            if valid and area > max_area:
                max_area = area
        
        return max_area if max_area != -1 else -1