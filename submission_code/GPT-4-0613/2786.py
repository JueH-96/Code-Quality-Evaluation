class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        dp = [[0]*n for _ in range(n)]
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                if s[i] == s[j] and (j-i == 1 or dp[i+1][j-1] == 1):
                    dp[i][j] = 1
                elif s[i] == s[j] and j-i > 1 and dp[i+1][j-1] > 1:
                    dp[i][j] = 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return max(max(dp[i]) for i in range(n))