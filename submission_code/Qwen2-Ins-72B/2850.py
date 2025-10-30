class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        if x == y:
            return 2 * (2 * x + z)
        elif x > y:
            return 2 * (2 * y + 1 + z)
        else:
            return 2 * (2 * x + 1 + z)