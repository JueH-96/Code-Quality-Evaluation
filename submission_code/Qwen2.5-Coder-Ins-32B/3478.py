from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if a point (x, y) is inside or on the boundary of any circle
        def is_inside_any_circle(x, y):
            for cx, cy, r in circles:
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    return True
            return False
        
        # Check if the path from (0, 0) to (xCorner, yCorner) can be blocked by circles
        # We only need to check the boundary points of the rectangle
        # and the points where the circles might intersect the rectangle
        
        # Check the four corners of the rectangle
        if is_inside_any_circle(0, 0) or is_inside_any_circle(xCorner, yCorner):
            return False
        
        # Check the points along the edges of the rectangle
        # For each circle, check if it intersects the edges of the rectangle
        for cx, cy, r in circles:
            # Check left edge (x = 0)
            if 0 <= cy <= yCorner and abs(cx) <= r:
                return False
            # Check right edge (x = xCorner)
            if 0 <= cy <= yCorner and abs(cx - xCorner) <= r:
                return False
            # Check bottom edge (y = 0)
            if 0 <= cx <= xCorner and abs(cy) <= r:
                return False
            # Check top edge (y = yCorner)
            if 0 <= cx <= xCorner and abs(cy - yCorner) <= r:
                return False
        
        # If none of the above conditions are met, there is a path
        return True