class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        inf = float('inf')
        dp = [[inf] * (n + 1) for _ in range(2)]
        dp[0][0] = dp[1][0] = 0

        for i in range(n):
            for j in range(2):
                if j == 1 and s1[i] == s2[i]:
                    dp[j][i + 1] = min(dp[j][i + 1], dp[j][i])
                if s1[i] != s2[i]:
                    dp[j][i + 1] = min(dp[j][i + 1], dp[j][i] + x)
                if i + 1 < n:
                    dp[1 - j][i + 2] = min(dp[1 - j][i + 2], dp[j][i + 1])

        return min(dp[0][n], dp[1][n]) if dp[0][n] < inf or dp[1][n] < inf else -1