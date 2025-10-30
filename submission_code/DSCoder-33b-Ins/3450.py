class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        if k % (2 * (n - 1)) < n:
            return k % (2 * (n - 1))
        else:
            return 2 * (n - 1) - k % (2 * (n - 1))