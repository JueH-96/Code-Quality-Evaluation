class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def intersects(x, y, r, x1, y1, x2, y2):
            dist = abs(x - x1) if y1 <= y <= y2 else float('inf')
            dist = min(dist, abs(x - x2) if y1 <= y <= y2 else float('inf'))
            dist = min(dist, abs(y - y1) if x1 <= x <= x2 else float('inf'))
            dist = min(dist, abs(y - y2) if x1 <= x <= x2 else float('inf'))
            
            return dist <= r

        for x, y, r in circles:
            if intersects(x, y, r, 0, 0, xCorner, 0) or \
               intersects(x, y, r, 0, 0, 0, yCorner) or \
               intersects(x, y, r, xCorner, yCorner, xCorner, 0) or \
               intersects(x, y, r, xCorner, yCorner, 0, yCorner):
                return False
        return True