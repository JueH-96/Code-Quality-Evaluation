class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        a = [1]*n
        MOD = 10**9 + 7
        for _ in range(k):
            for i in range(n-1, -1, -1):
                a[i] = (a[i] + a[i+1]) % MOD if i+1 < n else a[i]
        return a[0]