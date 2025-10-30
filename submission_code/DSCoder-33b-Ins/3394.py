class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        else:
            return (x + n - 1) & -n