class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        T = 2 * (n - 1)
        r = k % T if T != 0 else 0
        if r < n:
            return r
        else:
            return T - r