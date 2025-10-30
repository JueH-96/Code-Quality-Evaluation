class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        cycle = 2 * (n - 1)
        t = k % cycle
        if t < n - 1:
            return t
        else:
            return 2 * (n - 1) - t - 1