from typing import List
import math

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_blocked(x1, y1, x2, y2, x, y, r):
            # Check if the line segment (x1, y1) to (x2, y2) is blocked by the circle (x, y, r)
            # Distance from the center of the circle to the line segment
            dx = x2 - x1
            dy = y2 - y1
            dr2 = dx**2 + dy**2
            D = x1 * y2 - x2 * y1
            det = (r**2) * dr2 - (dx * (x - x1) + dy * (y - y1))**2
            if det < 0:
                return False
            else:
                det = math.sqrt(det)
                x3 = (dy * (dy * (x - x1) - dx * (y - y1)) - dx * D) / dr2
                y3 = (-dx * (dy * (x - x1) - dx * (y - y1)) - dy * D) / dr2
                if (x3 >= min(x1, x2) and x3 <= max(x1, x2) and y3 >= min(y1, y2) and y3 <= max(y1, y2)):
                    return True
                else:
                    return False

        for circle in circles:
            x, y, r = circle
            if (is_blocked(0, 0, xCorner, 0, x, y, r) or
                is_blocked(xCorner, 0, xCorner, yCorner, x, y, r) or
                is_blocked(0, 0, 0, yCorner, x, y, r) or
                is_blocked(0, yCorner, xCorner, yCorner, x, y, r)):
                return False
        return True