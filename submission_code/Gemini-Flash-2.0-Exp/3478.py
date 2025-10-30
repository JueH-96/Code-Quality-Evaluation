class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_safe(x, y):
            if x < 0 or x > xCorner or y < 0 or y > yCorner:
                return False
            for cx, cy, r in circles:
                if (x - cx)**2 + (y - cy)**2 <= r**2:
                    return False
            return True

        def can_reach(start_x, start_y, end_x, end_y, visited):
            if start_x == end_x and start_y == end_y:
                return True
            
            visited.add((start_x, start_y))

            # Possible moves (right, up)
            moves = [(start_x + 1, start_y), (start_x, start_y + 1)]

            for next_x, next_y in moves:
                if is_safe(next_x, next_y) and (next_x, next_y) not in visited:
                    if can_reach(next_x, next_y, end_x, end_y, visited):
                        return True
            
            return False

        # Check if start or end is inside any circle
        if not is_safe(0, 0) or not is_safe(xCorner, yCorner):
            return False

        # Check if a direct path is blocked
        def is_direct_path_blocked():
            for cx, cy, r in circles:
                # Check if the circle intersects the line segment from (0,0) to (xCorner, yCorner)
                # Parametric equation of the line: x = 0 + t * xCorner, y = 0 + t * yCorner, 0 <= t <= 1
                # Distance from circle center to the line: |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
                distance = abs((yCorner - 0) * cx - (xCorner - 0) * cy + xCorner * 0 - yCorner * 0) / ((yCorner - 0)**2 + (xCorner - 0)**2)**0.5
                if distance <= r:
                    # Check if the point on the line closest to the circle is within the segment
                    t = (cx * xCorner + cy * yCorner) / (xCorner**2 + yCorner**2)
                    if 0 < t < 1:
                        return True
            return False
        
        if not is_direct_path_blocked():
            return True

        # Use BFS to find a path
        visited = set()
        return can_reach(0, 0, xCorner, yCorner, visited)