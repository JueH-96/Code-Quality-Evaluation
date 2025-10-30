class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        period = 2 * (n - 1)
        rem = k % period
        if rem <= n - 1:
            return rem
        else:
            return period - rem