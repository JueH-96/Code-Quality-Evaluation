import math

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: list[list[int]]) -> bool:
        
        def is_safe(x, y):
            if x < 0 or x > xCorner or y < 0 or y > yCorner:
                return False
            for cx, cy, r in circles:
                dist_sq = (x - cx)**2 + (y - cy)**2
                if dist_sq <= r**2:
                    return False
            return True

        if not is_safe(0,0) or not is_safe(xCorner, yCorner):
            return False

        q = [(0, 0)]
        visited = {(0, 0)}
        
        while q:
            x, y = q.pop(0)
            if x == xCorner and y == yCorner:
                return True

            # Check adjacent points
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if is_safe(nx, ny) and (nx, ny) not in visited:
                    q.append((nx, ny))
                    visited.add((nx, ny))
        
        return False