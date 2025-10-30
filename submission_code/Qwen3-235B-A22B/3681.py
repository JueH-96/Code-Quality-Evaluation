class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        from typing import List

        # Convert points to a set of tuples for O(1) lookup
        points_set = set((x, y) for x, y in points)
        max_area = -1
        
        # Extract and sort all unique x and y coordinates
        x_coords = sorted({x for x, y in points})
        y_coords = sorted({y for y, x in points})
        
        # Iterate over all possible pairs of x coordinates (x1 < x2)
        for i in range(len(x_coords)):
            for j in range(i + 1, len(x_coords)):
                x1, x2 = x_coords[i], x_coords[j]
                # Iterate over all possible pairs of y coordinates (y1 < y2)
                for k in range(len(y_coords)):
                    for l in range(k + 1, len(y_coords)):
                        y1, y2 = y_coords[k], y_coords[l]
                        # Check if all four corners exist
                        if ((x1, y1) in points_set and 
                            (x1, y2) in points_set and 
                            (x2, y1) in points_set and 
                            (x2, y2) in points_set):
                            # Check all points to ensure none are inside or on the rectangle
                            valid = True
                            for px, py in points:
                                # Skip the four corners
                                if ((px == x1 and py == y1) or 
                                    (px == x1 and py == y2) or 
                                    (px == x2 and py == y1) or 
                                    (px == x2 and py == y2)):
                                    continue
                                # Check if point is inside or on the rectangle
                                if (x1 <= px <= x2) and (y1 <= py <= y2):
                                    valid = False
                                    break
                            if valid:
                                area = (x2 - x1) * (y2 - y1)
                                if area > max_area:
                                    max_area = area
        return max_area