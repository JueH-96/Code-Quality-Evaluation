from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        point_set = set(tuple(point) for point in points)
        max_area = -1
        n = len(points)
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                
                # Check if p1 and p2 can be diagonal corners
                if p1[0] != p2[0] and p1[1] != p2[1]:
                    # Define the other two corners
                    p3 = [p1[0], p2[1]]
                    p4 = [p2[0], p1[1]]
                    
                    if tuple(p3) in point_set and tuple(p4) in point_set:
                        # Determine the bounds of the rectangle
                        x_min = min(p1[0], p2[0])
                        x_max = max(p1[0], p2[0])
                        y_min = min(p1[1], p2[1])
                        y_max = max(p1[1], p2[1])
                        
                        # Check for any other point inside or on the border
                        valid = True
                        for p in points:
                            if tuple(p) in {(tuple(p1), tuple(p2), tuple(p3), tuple(p4))}:
                                continue
                            if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max:
                                valid = False
                                break
                        
                        if valid:
                            area = abs(x_max - x_min) * abs(y_max - y_min)
                            max_area = max(max_area, area)
        
        return max_area