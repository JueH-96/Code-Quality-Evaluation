class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        for x, y, r in circles:
            if not (x - r > xCorner or x + r < 0 or y - r > yCorner or y + r < 0):
                if (x - r <= 0 and x + r >= xCorner and y -r <=0 and y + r >= yCorner):
                    return False
                
                
                
        return True