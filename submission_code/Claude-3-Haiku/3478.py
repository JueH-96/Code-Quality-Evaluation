class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if the rectangle is completely outside all the circles
        for x, y, r in circles:
            if (x - r) <= xCorner and (y - r) <= yCorner and (x + r) >= 0 and (y + r) >= 0:
                return False
        
        # Check if the diagonal path from (0, 0) to (xCorner, yCorner) is valid
        for x, y, r in circles:
            if (x - r) <= xCorner and (y - r) <= yCorner and (x + r) >= 0 and (y + r) >= 0:
                if ((x - 0) ** 2 + (y - 0) ** 2) ** 0.5 <= r or ((x - xCorner) ** 2 + (y - yCorner) ** 2) ** 0.5 <= r:
                    return False
        
        return True