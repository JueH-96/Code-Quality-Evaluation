from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        from math import sqrt
        
        def isInsideCircle(x, y, circle):
            cx, cy, r = circle
            return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        
        def isTouchingCircle(x, y, circle):
            cx, cy, r = circle
            return (x - cx) ** 2 + (y - cy) ** 2 == r ** 2
        
        def isOnCircle(x, y, circle):
            return isInsideCircle(x, y, circle) or isTouchingCircle(x, y, circle)
        
        def isPathClear(x1, y1, x2, y2, circle):
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                y = y1
                while y <= y2:
                    if isOnCircle(x1, y, circle):
                        return False
                    y += 1
            elif dy == 0:
                x = x1
                while x <= x2:
                    if isOnCircle(x, y1, circle):
                        return False
                    x += 1
            else:
                slope = dy / dx
                intercept = y1 - slope * x1
                x = x1
                while x <= x2:
                    y = slope * x + intercept
                    if isOnCircle(x, y, circle):
                        return False
                    x += 1
            return True
        
        for i in range(xCorner):
            if isPathClear(0, i, xCorner, i, circles) or isPathClear(0, i, i, yCorner, circles):
                return True
            if isPathClear(i, 0, i, yCorner, circles) or isPathClear(i, 0, xCorner, 0, circles):
                return True
        
        return False