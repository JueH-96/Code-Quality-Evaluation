from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        if len(points) < 4:
            return -1
        
        # Extract unique x and y coordinates and sort them
        x_values = sorted(set(point[0] for point in points))
        y_values = sorted(set(point[1] for point in points))
        
        if len(x_values) < 2 or len(y_values) < 2:
            return -1  # Not enough unique x or y values to form a rectangle
        
        # Store points in a set for O(1) lookups
        point_set = set(tuple(point) for point in points)
        
        max_area = -1
        
        # Iterate through all pairs of x_values
        for i in range(len(x_values)):
            for j in range(i + 1, len(x_values)):
                x1 = x_values[i]
                x2 = x_values[j]
                # Iterate through all pairs of y_values
                for k in range(len(y_values)):
                    for l in range(k + 1, len(y_values)):
                        y1 = y_values[k]
                        y2 = y_values[l]
                        # Check if all four corners exist in the point set
                        if (x1, y1) in point_set and (x1, y2) in point_set and (x2, y1) in point_set and (x2, y2) in point_set:
                            # Check if no other points lie inside or on the border
                            valid = True
                            for point in points:
                                if point == [x1, y1] or point == [x1, y2] or point == [x2, y1] or point == [x2, y2]:
                                    continue
                                if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                                    valid = False
                                    break
                            if valid:
                                area = (x2 - x1) * (y2 - y1)
                                if area > max_area:
                                    max_area = area
        return max_area