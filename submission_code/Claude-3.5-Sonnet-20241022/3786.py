class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        
        # dp[i][j][k] represents the longest palindromic subsequence length
        # for substring s[i:j+1] with k operations remaining
        dp = [[[0] * (k + 1) for _ in range(n)] for _ in range(n)]
        
        # Initialize single characters (base case)
        for i in range(n):
            for ops in range(k + 1):
                dp[i][i][ops] = 1
        
        # Fill the dp table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for ops in range(k + 1):
                    # Case 1: Don't include either character
                    dp[i][j][ops] = max(dp[i+1][j][ops], dp[i][j-1][ops])
                    
                    if ops > 0:
                        # Try all possible character changes for s[i] and s[j]
                        for c1 in (chr((ord(s[i]) - 1 - ord('a')) % 26 + ord('a')),
                                 s[i],
                                 chr((ord(s[i]) + 1 - ord('a')) % 26 + ord('a'))):
                            for c2 in (chr((ord(s[j]) - 1 - ord('a')) % 26 + ord('a')),
                                     s[j],
                                     chr((ord(s[j]) + 1 - ord('a')) % 26 + ord('a'))):
                                cost = (c1 != s[i]) + (c2 != s[j])
                                if cost <= ops and c1 == c2:
                                    # If characters match after changes
                                    if i + 1 <= j - 1:
                                        dp[i][j][ops] = max(dp[i][j][ops],
                                                          dp[i+1][j-1][ops-cost] + 2)
                                    else:
                                        dp[i][j][ops] = max(dp[i][j][ops], 2)
                    elif s[i] == s[j]:
                        # If characters already match without changes
                        if i + 1 <= j - 1:
                            dp[i][j][ops] = max(dp[i][j][ops],
                                              dp[i+1][j-1][ops] + 2)
                        else:
                            dp[i][j][ops] = max(dp[i][j][ops], 2)
        
        return dp[0][n-1][k]