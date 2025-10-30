class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        min_xy = min(x, y)
        length = 2 * (2 * min_xy + z + (1 if x != y else 0))
        return length