class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        C = [[0] * (n + k + 1) for _ in range(n + k + 1)]
        for i in range(n + k + 1):
            C[i][0] = 1
            for j in range(1, i + 1):
                C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD
        return C[n + k - 1][n - 1]