class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        
        # dp[i][j] will hold the maximum length of palindromic subsequence
        # in s[i:j+1] with at most k changes allowed
        dp = [[[0] * (k + 1) for _ in range(n)] for _ in range(n)]
        
        # Base case: single letters are palindromes of length 1
        for i in range(n):
            for j in range(k + 1):
                dp[i][i][j] = 1
        
        # Fill the dp table
        for length in range(2, n + 1):  # length of the substring
            for i in range(n - length + 1):
                j = i + length - 1
                for changes in range(k + 1):
                    if s[i] == s[j]:
                        dp[i][j][changes] = dp[i + 1][j - 1][changes] + 2
                    else:
                        dp[i][j][changes] = max(dp[i + 1][j][changes], dp[i][j - 1][changes])
                        if changes > 0:
                            # Try changing s[i] or s[j]
                            dp[i][j][changes] = max(dp[i][j][changes], dp[i + 1][j - 1][changes - 1] + 2)
                            dp[i][j][changes] = max(dp[i][j][changes], dp[i + 1][j][changes - 1])
                            dp[i][j][changes] = max(dp[i][j][changes], dp[i][j - 1][changes - 1])
        
        return dp[0][n - 1][k]