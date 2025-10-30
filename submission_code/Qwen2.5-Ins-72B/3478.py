from typing import List
import math

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        def isBlocked(x1, y1, x2, y2, circles):
            for x, y, r in circles:
                if distance(x1, y1, x, y) <= r or distance(x2, y2, x, y) <= r:
                    return True
                if distance(x1, y1, x, y) + distance(x2, y2, x, y) <= r:
                    return True
                if (x1 <= x <= x2 and y1 <= y <= y2) or (x1 <= x <= x2 and y2 <= y <= y1) or (x2 <= x <= x1 and y1 <= y <= y2) or (x2 <= x <= x1 and y2 <= y <= y1):
                    if distance(x1, y1, x, y) + distance(x2, y2, x, y) <= r:
                        return True
            return False
        
        def isPathPossible(x1, y1, x2, y2, circles):
            if isBlocked(x1, y1, x2, y2, circles):
                return False
            if x1 == x2:
                for y in range(min(y1, y2) + 1, max(y1, y2)):
                    if isBlocked(x1, y1, x1, y, circles):
                        return False
            elif y1 == y2:
                for x in range(min(x1, x2) + 1, max(x1, x2)):
                    if isBlocked(x1, y1, x, y2, circles):
                        return False
            else:
                for x in range(min(x1, x2) + 1, max(x1, x2)):
                    y = y1 + (y2 - y1) * (x - x1) // (x2 - x1)
                    if isBlocked(x1, y1, x, y, circles):
                        return False
            return True
        
        return isPathPossible(0, 0, xCorner, yCorner, circles)