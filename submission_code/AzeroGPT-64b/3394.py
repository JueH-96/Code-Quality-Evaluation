class Solution:
    def minEnd(self, n: int, x: int) -> int:
        for i in range(31):
            if x & (1 << i) == 0:
                x |= (n - 1) << i
        return x + n - 1