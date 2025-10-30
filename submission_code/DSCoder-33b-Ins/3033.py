class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        max_len = 0
        for i in range(n):
            for j in range(n):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                    max_len = max(max_len, dp[i + 1][j + 1])
        if max_len >= n - max_len:
            return (n - max_len) * x
        else:
            return -1