class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if the rectangle's corners are inside any circle
        for circle in circles:
            if (circle[0]-circle[2]) <= 0 and (circle[1]-circle[2]) <= 0 and (circle[0]+circle[2]) >= xCorner and (circle[1]+circle[2]) >= yCorner:
                return False
        
        # Check if the rectangle's corners are inside any circle
        for circle in circles:
            if (circle[0]+circle[2]) >= xCorner and (circle[1]-circle[2]) <= 0 and (circle[0]-circle[2]) <= 0 and (circle[1]+circle[2]) >= yCorner:
                return False
        
        # Check if the rectangle's corners are inside any circle
        for circle in circles:
            if (circle[0]-circle[2]) <= 0 and (circle[1]+circle[2]) >= yCorner and (circle[0]+circle[2]) >= xCorner and (circle[1]-circle[2]) <= 0:
                return False
        
        # Check if the rectangle's corners are inside any circle
        for circle in circles:
            if (circle[0]+circle[2]) >= xCorner and (circle[1]+circle[2]) >= yCorner and (circle[0]-circle[2]) <= 0 and (circle[1]-circle[2]) <= 0:
                return False
        
        return True