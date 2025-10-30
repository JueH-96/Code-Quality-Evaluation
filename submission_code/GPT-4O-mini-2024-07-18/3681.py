from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        point_set = set(map(tuple, points))
        max_area = -1
        
        # Check all combinations of points to form rectangles
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                # Get the two corners of the rectangle
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                # Ensure we have opposite corners
                if x1 != x2 and y1 != y2:
                    # The other two corners of the rectangle
                    p3 = (x1, y2)
                    p4 = (x2, y1)
                    
                    # Check if the other two corners are in the set
                    if p3 in point_set and p4 in point_set:
                        # Calculate area
                        area = abs(x2 - x1) * abs(y2 - y1)
                        
                        # Check if there are any points inside the rectangle
                        has_inside_point = False
                        for x, y in points:
                            if (x1 < x < x2) and (y1 < y < y2):
                                has_inside_point = True
                                break
                        
                        if not has_inside_point:
                            max_area = max(max_area, area)
        
        return max_area