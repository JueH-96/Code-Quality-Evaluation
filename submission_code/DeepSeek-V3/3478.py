from typing import List
import math

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if any circle completely blocks the rectangle
        for circle in circles:
            x, y, r = circle
            # Check if the circle covers the entire rectangle
            if x - r <= 0 and x + r >= xCorner and y - r <= 0 and y + r >= yCorner:
                return False
        return True