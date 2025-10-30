class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if start or end points are inside any circle
        for x, y, r in circles:
            # Check start point (0,0)
            if x*x + y*y <= r*r:
                return False
            # Check end point (xCorner,yCorner)
            if (x-xCorner)*(x-xCorner) + (y-yCorner)*(y-yCorner) <= r*r:
                return False
        
        # For each circle, check if it blocks the line from (0,0) to (xCorner,yCorner)
        for x, y, r in circles:
            # Calculate shortest distance from circle center to line segment
            # Using point-line distance formula: |ax + by + c|/sqrt(a^2 + b^2)
            # Line equation: y = (yCorner/xCorner)x
            # Rearranging to ax + by + c = 0 form: (yCorner)x - (xCorner)y = 0
            
            a = yCorner
            b = -xCorner
            c = 0
            
            dist = abs(a*x + b*y + c) / ((a*a + b*b) ** 0.5)
            
            # If circle center is within rectangle bounds and distance is <= radius
            # then circle blocks the path
            if (0 <= x <= xCorner and 0 <= y <= yCorner and dist <= r):
                return False
                
            # Check if circle intersects with line segment
            # Project circle center onto line
            t = -(a*x + b*y + c)/(a*a + b*b)
            px = x + t*a
            py = y + t*b
            
            # Check if projection point lies on line segment
            if (0 <= px <= xCorner and 0 <= py <= yCorner):
                if dist <= r:
                    return False
        
        return True