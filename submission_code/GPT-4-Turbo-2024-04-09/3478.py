class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        from math import sqrt
        
        def intersects_or_inside(x, y, r, x1, y1, x2, y2):
            # Check if circle with center (x, y) and radius r intersects or is inside the rectangle
            # defined by corners (x1, y1) and (x2, y2)
            
            # Find the nearest point on the rectangle to the circle center
            nearest_x = max(x1, min(x, x2))
            nearest_y = max(y1, min(y, y2))
            
            # Calculate distance from nearest point to the circle center
            distance = sqrt((nearest_x - x) ** 2 + (nearest_y - y) ** 2)
            
            # If distance is less than or equal to radius, circle intersects or is inside the rectangle
            return distance <= r
        
        # Check if any circle blocks the entire width or height of the rectangle
        for x, y, r in circles:
            if intersects_or_inside(x, y, r, 0, 0, xCorner, 0) and intersects_or_inside(x, y, r, 0, yCorner, xCorner, yCorner):
                return False
            if intersects_or_inside(x, y, r, 0, 0, 0, yCorner) and intersects_or_inside(x, y, r, xCorner, 0, xCorner, yCorner):
                return False
        
        return True