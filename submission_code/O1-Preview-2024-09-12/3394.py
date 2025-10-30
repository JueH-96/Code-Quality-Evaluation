class Solution:
    def minEnd(self, n: int, x: int) -> int:
        b = 0
        while (x >> b) & 1:
            b += 1
        return x + (n - 1) * (1 << b)