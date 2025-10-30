from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert list of points to a set of tuples for fast membership tests
        point_set = {tuple(p) for p in points}
        
        # Get all unique x and y coordinates from the points
        xs = sorted({p[0] for p in points})
        ys = sorted({p[1] for p in points})
        
        max_area = -1
        
        # Iterate over all pairs of distinct x-coordinates and y-coordinates
        # These will define the potential rectangle boundaries.
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                x1, x2 = xs[i], xs[j]
                for k in range(len(ys)):
                    for l in range(k+1, len(ys)):
                        y1, y2 = ys[k], ys[l]
                        # For a valid rectangle, the four corner points must exist.
                        corners = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}
                        if not corners.issubset(point_set):
                            continue
                        
                        # Check that no extra points lie inside or on the border.
                        valid = True
                        for p in point_set:
                            # Skip the corners that we are using for the rectangle.
                            if p in corners:
                                continue
                            # If any other point is inside or on the border, discard this rectangle.
                            if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                                valid = False
                                break
                                
                        if valid:
                            area = (x2 - x1) * (y2 - y1)
                            max_area = max(max_area, area)
        
        return max_area