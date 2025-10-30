class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        dp = [[0]*(n+1) for _ in range(n+1)]
        for i in range(n+1):
            for j in range(n+1):
                if i == 0 and j == 0:
                    continue
                if i == 0 or (j > 0 and dp[i][j-1] < dp[i-1][j]):
                    dp[i][j] = dp[i][j-1] + (s1[j-1] != s2[j-1])
                else:
                    dp[i][j] = dp[i-1][j] + (s1[i-1] != s2[i-1])
        res = float('inf')
        for i in range(n+1):
            for j in range(n+1):
                if dp[i][j] <= x:
                    res = min(res, dp[i][j] + abs(i-j))
        return res if res != float('inf') else -1