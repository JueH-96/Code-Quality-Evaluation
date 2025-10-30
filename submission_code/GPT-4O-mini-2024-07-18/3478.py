from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_inside_circle(x, y, circle):
            cx, cy, r = circle
            return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
        
        # Check if the corners are inside any circle
        if any(is_inside_circle(0, 0, circle) for circle in circles) or any(is_inside_circle(xCorner, yCorner, circle) for circle in circles):
            return False
        
        # Check if the path can be made without touching any circle
        # We will check the four edges of the rectangle
        for circle in circles:
            cx, cy, r = circle
            
            # Check if the circle intersects with the left edge (x = 0)
            if 0 <= cx <= 0 + r and 0 <= cy <= yCorner:
                return False
            
            # Check if the circle intersects with the right edge (x = xCorner)
            if xCorner - r <= cx <= xCorner and 0 <= cy <= yCorner:
                return False
            
            # Check if the circle intersects with the bottom edge (y = 0)
            if 0 <= cx <= xCorner and 0 <= cy <= 0 + r:
                return False
            
            # Check if the circle intersects with the top edge (y = yCorner)
            if 0 <= cx <= xCorner and yCorner - r <= cy <= yCorner:
                return False
        
        return True