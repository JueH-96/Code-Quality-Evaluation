class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        for cx, cy, r in circles:
            # Check if either endpoint is inside the circle
            if cx**2 + cy**2 <= r**2 or (cx - xCorner)**2 + (cy - yCorner)**2 <= r**2:
                return False
            
            # Calculate the parameter t for the closest point on the line segment to the center of the circle
            dot = cx * xCorner + cy * yCorner
            len_sq = xCorner**2 + yCorner**2
            t = max(0, min(1, dot / len_sq))
            
            # Calculate the closest point on the line segment to the center of the circle
            x_closest = t * xCorner
            y_closest = t * yCorner
            
            # Calculate the squared distance from the center of the circle to the closest point
            distance_sq = (cx - x_closest)**2 + (cy - y_closest)**2
            
            # If the squared distance is less than the squared radius of the circle,
            # then the line segment passes through the circle
            if distance_sq < r**2:
                return False
        
        return True