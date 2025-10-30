class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        if n == 0:
            return 0
        # Initialize the 3D DP array
        dp = [[[0]*(k+1) for _ in range(n)] for _ in range(n)]
        
        # Base case: substrings of length 1
        for i in range(n):
            for op in range(k+1):
                dp[i][i][op] = 1
        
        # Fill the DP table for substrings of length >= 2
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                for op in range(k+1):
                    if s[i] == s[j]:
                        dp[i][j][op] = 2 + dp[i+1][j-1][op]
                    else:
                        # Calculate the minimal cost to make s[i] and s[j] the same
                        cost_diff = abs(ord(s[i]) - ord(s[j]))
                        cost = min(cost_diff, 26 - cost_diff)
                        # Option 1: do not use one of the characters
                        option1 = max(dp[i+1][j][op], dp[i][j-1][op])
                        # Option 2: use both characters if possible
                        option2 = 0
                        if op >= cost:
                            option2 = 2 + dp[i+1][j-1][op - cost]
                        dp[i][j][op] = max(option1, option2)
        return dp[0][n-1][k]