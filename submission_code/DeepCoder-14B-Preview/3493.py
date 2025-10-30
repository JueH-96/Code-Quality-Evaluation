class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if (0,0) is inside any circle
        for x, y, r in circles:
            dx = x
            dy = y
            if dx * dx + dy * dy <= r * r:
                return False
        
        # Check if (xCorner, yCorner) is inside any circle
        for x, y, r in circles:
            dx = x - xCorner
            dy = y - yCorner
            if dx * dx + dy * dy <= r * r:
                return False
        
        # If neither point is inside any circle, assume a path exists
        return True