class Solution:
    def minEnd(self, n: int, x: int) -> int:
        return x | ((1 << n) - 1)