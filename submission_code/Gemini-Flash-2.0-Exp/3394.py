class Solution:
    def minEnd(self, n: int, x: int) -> int:
        res = x
        for i in range(1, n):
            res |= (1 << i)
        return res