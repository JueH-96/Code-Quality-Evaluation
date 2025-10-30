class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def dist(x1, y1, x2, y2):
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        def check(x, y):
            for cx, cy, r in circles:
                if dist(x, y, cx, cy) <= r:
                    return False
            return True

        if not check(0, 0) or not check(xCorner, yCorner):
            return False

        for cx, cy, r in circles:
            if dist(0, 0, cx, cy) <= r or dist(xCorner, yCorner, cx, cy) <= r:
                return False

        for cx, cy, r in circles:
            if cy <= r and check(0, cy + r) and check(cx + r, yCorner):
                return True
            if cx <= r and check(cx + r, 0) and check(xCorner, cy + r):
                return True

        return False