class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        if x == y:
            return (x + y + z) * 2
        else:
            return (min(x, y) * 2 + 1 + z) * 2