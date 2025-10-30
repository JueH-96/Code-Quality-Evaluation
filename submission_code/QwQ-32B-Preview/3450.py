class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        period = 2 * (n - 1)
        t = k % period
        if t < n:
            return t
        else:
            return 2 * (n - 1) - t