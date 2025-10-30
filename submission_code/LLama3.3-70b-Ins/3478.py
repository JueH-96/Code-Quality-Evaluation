from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def isBlocked(x: int, y: int) -> bool:
            for circle in circles:
                if (x - circle[0]) ** 2 + (y - circle[1]) ** 2 <= circle[2] ** 2:
                    return True
            return False

        if isBlocked(0, 0) or isBlocked(xCorner, yCorner):
            return False

        for circle in circles:
            if (circle[0] - circle[2] <= 0 and circle[0] + circle[2] >= xCorner and 
                circle[1] - circle[2] <= 0 and circle[1] + circle[2] >= yCorner):
                return False

        return True