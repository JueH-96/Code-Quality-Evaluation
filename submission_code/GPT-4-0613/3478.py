class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: [[int]]) -> bool:
        for x, y, r in circles:
            if x < r or y < r or xCorner - x < r or yCorner - y < r:
                return False
        return True