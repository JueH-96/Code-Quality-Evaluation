class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        min_xy = min(x, y)
        extra = 2 if x != y else 0
        return 2 * (2 * min_xy + z) + extra