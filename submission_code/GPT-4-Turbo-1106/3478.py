from typing import List
import math

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_inside_circle(x, y, circle):
            cx, cy, r = circle
            return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
        
        def does_intersect(x1, y1, x2, y2, circle):
            cx, cy, r = circle
            # Line equation coefficients A*x + B*y + C = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            # Distance from circle center to line
            dist = abs(A * cx + B * cy + C) / math.sqrt(A ** 2 + B ** 2)
            # Check if the distance is less than the radius and circle is between the line segment endpoints
            if dist < r:
                dot1 = (cx - x1) * (x2 - x1) + (cy - y1) * (y2 - y1)
                dot2 = (cx - x2) * (x1 - x2) + (cy - y2) * (y1 - y2)
                return dot1 > 0 and dot2 > 0
            return False
        
        for circle in circles:
            # Check if the circle is inside the rectangle
            if is_inside_circle(0, 0, circle) or is_inside_circle(xCorner, yCorner, circle):
                return False
            # Check if the circle intersects with the two diagonal lines of the rectangle
            if does_intersect(0, 0, xCorner, yCorner, circle) or does_intersect(0, yCorner, xCorner, 0, circle):
                return False
        
        return True

# Example usage:
# sol = Solution()
# print(sol.canReachCorner(3, 4, [[2,1,1]])) # Output: true
# print(sol.canReachCorner(3, 3, [[1,1,2]])) # Output: false
# print(sol.canReachCorner(3, 3, [[2,1,1],[1,2,1]])) # Output: false
# print(sol.canReachCorner(4, 4, [[5,5,1]])) # Output: true