class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def intersects_circle(x1, y1, x2, y2, cx, cy, r):
            # Check if line segment (x1,y1)-(x2,y2) intersects circle (cx,cy,r)
            dx, dy = x2 - x1, y2 - y1
            a = dx**2 + dy**2
            b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
            c = cx**2 + cy**2 + x1**2 + y1**2 - 2 * (cx * x1 + cy * y1) - r**2
            
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return False
            
            t1 = (-b + discriminant**0.5) / (2 * a)
            t2 = (-b - discriminant**0.5) / (2 * a)
            
            return (0 <= t1 <= 1) or (0 <= t2 <= 1)
        
        def intersects_any_circle(x1, y1, x2, y2):
            return any(intersects_circle(x1, y1, x2, y2, cx, cy, r) for cx, cy, r in circles)
        
        # Check if start or end point is inside any circle
        if any((cx-r <= 0 <= cx+r and cy-r <= 0 <= cy+r) or 
               (cx-r <= xCorner <= cx+r and cy-r <= yCorner <= cy+r) 
               for cx, cy, r in circles):
            return False
        
        # Check if diagonal intersects any circle
        if intersects_any_circle(0, 0, xCorner, yCorner):
            return False
        
        # Check if path along edges intersects any circle
        if intersects_any_circle(0, 0, xCorner, 0) or intersects_any_circle(xCorner, 0, xCorner, yCorner):
            return False
        
        return True