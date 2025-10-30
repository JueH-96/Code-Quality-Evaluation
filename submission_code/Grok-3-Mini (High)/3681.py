from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        if len(points) < 4:
            return -1
        
        # Create a set for O(1) lookups
        point_set = set(tuple(p) for p in points)
        
        # Get sorted unique x and y coordinates
        x_coords = sorted(set(p[0] for p in points))
        y_coords = sorted(set(p[1] for p in points))
        
        max_area = -1
        
        # Iterate over all pairs of x coordinates
        for i in range(len(x_coords)):
            for j in range(i + 1, len(x_coords)):
                x_left = x_coords[i]
                x_right = x_coords[j]
                
                # Iterate over all pairs of y coordinates
                for k in range(len(y_coords)):
                    for l in range(k + 1, len(y_coords)):
                        y_bottom = y_coords[k]
                        y_top = y_coords[l]
                        
                        # Check if all four corners exist
                        if (x_left, y_bottom) in point_set and \
                           (x_left, y_top) in point_set and \
                           (x_right, y_bottom) in point_set and \
                           (x_right, y_top) in point_set:
                            
                            # Count points inside or on the border of the rectangle
                            count = sum(1 for p in points if x_left <= p[0] <= x_right and y_bottom <= p[1] <= y_top)
                            
                            # If exactly four points, it's a valid rectangle
                            if count == 4:
                                area = (x_right - x_left) * (y_top - y_bottom)
                                max_area = max(max_area, area)
        
        return max_area