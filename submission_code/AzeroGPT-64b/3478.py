class Solution:
   def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Function to check if the point crosses a certain circle.
        def crosses_circle(cx, cy, cr, px, py):
            return (cx-px)**2 + (cy-py)**2 <= cr**2

        # Check if safe to move right and then up.
        for x, y, r in circles:
            if x <= 0 and y >= yCorner and crosses_circle(x, y, r, 0, 0):
                return False
            if x >= xCorner and y >= yCorner and crosses_circle(x, y, r, xCorner, yCorner):
                return False
            if y <= 0 and x >= xCorner and crosses_circle(x, y, r, xCorner, 0):
                return False

        # Check if safe to move up and then right.
        for x, y, r in circles:
            if x <= 0 and y >= yCorner and crosses_circle(x, y, r, 0, 0):
                return False
            if x >= xCorner and y >= yCorner and crosses_circle(x, y, r, xCorner, yCorner):
                return False
            if y <= 0 and x <= 0 and crosses_circle(x, y, r, 0, 0):
                return False

        return True