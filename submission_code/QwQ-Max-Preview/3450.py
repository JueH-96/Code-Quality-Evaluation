class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        cycle_length = 2 * (n - 1)
        m = k % cycle_length
        if m <= n - 1:
            return m
        else:
            return 2 * (n - 1) - m