class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_point_inside_circle(x, y, circle):
            cx, cy, r = circle
            return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

        def is_path_blocked():
            # Check if the line from (0, 0) to (xCorner, yCorner) intersects any circle
            for circle in circles:
                cx, cy, r = circle
                # Calculate the distance from the line to the circle center
                # Line equation: y = (yCorner/xCorner) * x
                # Distance from point (cx, cy) to line y = mx is:
                # |m*cx - cy| / sqrt(m^2 + 1)
                m = yCorner / xCorner
                distance = abs(m * cx - cy) / (m ** 2 + 1) ** 0.5
                # Check if the circle intersects the line segment
                if distance <= r:
                    # Check if the perpendicular from the circle center to the line
                    # falls within the segment (0,0) to (xCorner, yCorner)
                    # Parametric line equation: (x, y) = t * (xCorner, yCorner)
                    # Find t such that the perpendicular from (cx, cy) to the line
                    # intersects the line at t * (xCorner, yCorner)
                    t = (cx + m * cy) / (xCorner + m * yCorner)
                    if 0 <= t <= 1:
                        return True
            return False

        return not is_path_blocked()