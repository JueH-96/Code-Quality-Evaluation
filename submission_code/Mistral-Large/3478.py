from typing import List
import math

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def distance_to_line(x0, y0, x1, y1, x2, y2):
            # Calculate the distance from point (x2, y2) to the line segment (x0, y0) to (x1, y1)
            num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            return num / den

        for x, y, r in circles:
            dist = distance_to_line(0, 0, xCorner, yCorner, x, y)
            if dist <= r:
                return False

        return True