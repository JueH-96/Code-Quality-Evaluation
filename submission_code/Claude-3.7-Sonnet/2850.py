class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        return 4 * min(x, y) + 2 * min(abs(x - y), z) + 2 * z