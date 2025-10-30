from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        """
        Finds the maximum area of an empty, axis-aligned rectangle formed by four given points.
        """
        n = len(points)
        
        # A rectangle requires at least 4 points.
        if n < 4:
            return -1

        # Use a set for efficient O(1) lookups of points.
        # Points are converted to tuples because lists are not hashable.
        point_set = set(tuple(p) for p in points)
        
        max_area = -1

        # Iterate through all unique pairs of points.
        # Consider each pair as a potential diagonal of a rectangle.
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]

                # A diagonal of a non-degenerate, axis-aligned rectangle
                # must connect points with different x and y coordinates.
                if x1 == x2 or y1 == y2:
                    continue

                # The other two corners would be (x1, y2) and (x2, y1).
                # Check if these points exist in our input set.
                if (x1, y2) in point_set and (x2, y1) in point_set:
                    # Found a potential rectangle.
                    # Define its bounding box.
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    
                    # The rectangle is "empty" if it contains no other points from the
                    # input list inside or on its border. This means the total number
                    # of points within its bounding box must be exactly 4 (the corners).
                    count = 0
                    for px, py in points:
                        if x_min <= px <= x_max and y_min <= py <= y_max:
                            count += 1
                    
                    if count == 4:
                        # This is a valid empty rectangle.
                        area = (x_max - x_min) * (y_max - y_min)
                        max_area = max(max_area, area)

        return max_area