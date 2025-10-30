class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        m = min(x, y)
        return 4 * m + 2 * z + (2 if x != y else 0)