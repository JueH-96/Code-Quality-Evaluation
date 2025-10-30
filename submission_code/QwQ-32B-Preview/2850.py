class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        return 2 * (x + min(y, x + 1) + z)