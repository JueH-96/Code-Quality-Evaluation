class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        if x == y:
            return (x + y) * 2 + 2 * z
        else:
            return (min(x, y) * 2 + 2 * z + 2)