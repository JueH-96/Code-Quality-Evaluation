class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        def min_ops_to_equal(c1, c2):
            if c1 == c2:
                return 0
            return min(abs(ord(c1) - ord(c2)), 26 - abs(ord(c1) - ord(c2)))
        
        n = len(s)
        # dp[i][j][ops] = max length of palindromic subsequence in s[i:j+1] with at most ops operations
        dp = [[[0] * (k + 1) for _ in range(n)] for _ in range(n)]
        
        # Base case: single characters
        for i in range(n):
            for ops in range(k + 1):
                dp[i][i][ops] = 1
        
        # Fill the DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for ops in range(k + 1):
                    # Option 1: don't include s[i]
                    dp[i][j][ops] = dp[i + 1][j][ops]
                    # Option 2: don't include s[j]
                    dp[i][j][ops] = max(dp[i][j][ops], dp[i][j - 1][ops])
                    # Option 3: include both s[i] and s[j]
                    cost = min_ops_to_equal(s[i], s[j])
                    if ops >= cost:
                        if i + 1 <= j - 1:
                            dp[i][j][ops] = max(dp[i][j][ops], 2 + dp[i + 1][j - 1][ops - cost])
                        else:
                            # i + 1 > j - 1, so the inner substring is empty
                            dp[i][j][ops] = max(dp[i][j][ops], 2)
        
        return dp[0][n - 1][k]