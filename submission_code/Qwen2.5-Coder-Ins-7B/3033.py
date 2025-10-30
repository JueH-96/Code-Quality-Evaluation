class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff = [i for i in range(n) if s1[i] != s2[i]]
        m = len(diff)
        if m % 2 != 0:
            return -1
        if m == 0:
            return 0
        dp = [[float('inf')] * (m + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = float('inf')
            dp[0][i] = float('inf')
        for i in range(1, m + 1):
            for j in range(1, i + 1):
                dp[i][j] = min(dp[i - 1][j] + x, dp[i - 2][j - 1] + 1)
        return dp[m][m // 2]