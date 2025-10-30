class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def distance(x1, y1, x2, y2):
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        def is_point_inside_circle(x, y, circle):
            cx, cy, r = circle
            return distance(x, y, cx, cy) <= r

        def is_path_blocked(x1, y1, x2, y2):
            for circle in circles:
                cx, cy, r = circle
                if distance(x1, y1, cx, cy) <= r and distance(x2, y2, cx, cy) <= r:
                    return True
            return False

        # Check if the starting point (0, 0) is inside any circle
        if any(is_point_inside_circle(0, 0, circle) for circle in circles):
            return False

        # Check if the ending point (xCorner, yCorner) is inside any circle
        if any(is_point_inside_circle(xCorner, yCorner, circle) for circle in circles):
            return False

        # Check if there is a direct path from (0, 0) to (xCorner, yCorner) that is blocked by any circle
        if is_path_blocked(0, 0, xCorner, yCorner):
            return False

        return True