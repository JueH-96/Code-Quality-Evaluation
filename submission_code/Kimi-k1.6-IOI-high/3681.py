import itertools
from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        points_set = set(tuple(point) for point in points)
        max_area = -1
        
        for p1, p2 in itertools.combinations(points, 2):
            x1, y1 = p1
            x2, y2 = p2
            
            if x1 == x2 or y1 == y2:
                continue
            
            p3 = (x1, y2)
            p4 = (x2, y1)
            
            if p3 in points_set and p4 in points_set:
                min_x = min(x1, x2)
                max_x = max(x1, x2)
                min_y = min(y1, y2)
                max_y = max(y1, y2)
                
                corners = { (x1, y1), (x2, y2), p3, p4 }
                valid = True
                
                for x, y in points:
                    if (x, y) not in corners:
                        if min_x <= x <= max_x and min_y <= y <= max_y:
                            valid = False
                            break
                
                if valid:
                    area = (max_x - min_x) * (max_y - min_y)
                    if area > max_area:
                        max_area = area
        
        return max_area if max_area != -1 else -1