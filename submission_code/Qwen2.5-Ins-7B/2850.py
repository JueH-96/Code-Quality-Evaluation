class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        if x == y:
            return 4 * x + 2 * z
        elif x < y:
            return 2 * (x + y + z) - 2
        else:
            return 2 * (x + y + z) - 2