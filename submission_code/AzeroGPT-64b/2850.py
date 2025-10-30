class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        if x == 0 and y == 0:
            return z
        if x == 0 or y == 0:
            f = max(x, y) + z
            return 2 * f
        m = min(x, y)
        f = 2 * (m + z) + 2
        d = max(x, y) - m
        if d * 2 > f:
            f = d * 2
        return f