from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        n = len(points)
        if n < 4:
            return -1
        
        point_set = set()
        for p in points:
            point_set.add((p[0], p[1]))
        
        max_area = -1
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                if x1 == x2 or y1 == y2:
                    continue
                if (x1, y2) not in point_set or (x2, y1) not in point_set:
                    continue
                
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                
                corners = {(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)}
                valid = True
                for p in point_set:
                    if p in corners:
                        continue
                    x, y = p
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        valid = False
                        break
                
                if valid:
                    area = (x_max - x_min) * (y_max - y_min)
                    if area > max_area:
                        max_area = area
        
        return max_area