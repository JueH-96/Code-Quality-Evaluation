class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        
        # dp[i][j][ops] = max length of palindromic subsequence in s[i:j+1] using at most ops operations
        dp = [[[0 for _ in range(k + 1)] for _ in range(n)] for _ in range(n)]
        
        # Base case: single characters
        for i in range(n):
            for ops in range(k + 1):
                dp[i][i][ops] = 1
        
        # Fill for all substring lengths
        for length in range(2, n + 1):  # substring length
            for i in range(n - length + 1):
                j = i + length - 1
                
                for ops in range(k + 1):
                    # Option 1: Don't include s[i] in the subsequence
                    if i + 1 <= j:
                        dp[i][j][ops] = max(dp[i][j][ops], dp[i + 1][j][ops])
                    
                    # Option 2: Don't include s[j] in the subsequence
                    if i <= j - 1:
                        dp[i][j][ops] = max(dp[i][j][ops], dp[i][j - 1][ops])
                    
                    # Option 3: Include both s[i] and s[j]
                    if i < j:
                        # Calculate minimum operations needed to make s[i] and s[j] the same
                        cost = min(abs(ord(s[i]) - ord(s[j])), 26 - abs(ord(s[i]) - ord(s[j])))
                        
                        if cost <= ops:
                            remaining_ops = ops - cost
                            if i + 1 <= j - 1:
                                dp[i][j][ops] = max(dp[i][j][ops], 2 + dp[i + 1][j - 1][remaining_ops])
                            else:
                                dp[i][j][ops] = max(dp[i][j][ops], 2)
                    
                    # Also consider using fewer operations
                    if ops > 0:
                        dp[i][j][ops] = max(dp[i][j][ops], dp[i][j][ops - 1])
        
        return dp[0][n - 1][k]