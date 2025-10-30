class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        mod = 10**9 + 7
        res = 1
        for i in range(1, k + 1):
            res = res * (n + i - 1) % mod
            res = res * pow(i, mod - 2, mod) % mod
        return res