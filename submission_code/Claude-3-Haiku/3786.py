class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = 1
        
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        
        max_length = dp[0][n-1]
        
        for i in range(n):
            for j in range(i, n):
                if abs(ord(s[i]) - ord(s[j])) <= k:
                    dp[i][j] = max(dp[i][j], dp[i+1][j-1] + 2)
                    max_length = max(max_length, dp[i][j])
        
        return max_length