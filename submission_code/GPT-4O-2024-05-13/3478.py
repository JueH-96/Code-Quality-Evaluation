from typing import List
import math

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_inside_circle(x, y, circle):
            cx, cy, r = circle
            return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        
        def is_path_blocked():
            for circle in circles:
                if is_inside_circle(0, 0, circle) or is_inside_circle(xCorner, yCorner, circle):
                    return True
            return False
        
        return not is_path_blocked()