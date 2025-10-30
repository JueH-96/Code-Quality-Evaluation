class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # Find all positions where strings differ
        diff = [i for i in range(len(s1)) if s1[i] != s2[i]]
        
        # If odd number of differences, impossible to make equal
        if len(diff) % 2 == 1:
            return -1
        
        n = len(diff)
        if n == 0:
            return 0
        
        if n == 2:
            return 1 if diff[1] == diff[0] + 1 else x
        
        # dp[i][j] = minimum cost to pair up diff[i:j+1]
        dp = [[float('inf')] * n for _ in range(n)]
        
        # Base case: pair adjacent elements in diff array
        for i in range(n - 1):
            cost = 1 if diff[i+1] == diff[i] + 1 else x
            dp[i][i+1] = cost
        
        # Fill DP table for larger ranges (only even lengths)
        for length in range(4, n + 1, 2):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Option 1: pair diff[i] with diff[j]
                cost = 1 if diff[j] == diff[i] + 1 else x
                dp[i][j] = min(dp[i][j], dp[i+1][j-1] + cost)
                
                # Option 2: split the range at position k
                for k in range(i + 2, j, 2):
                    dp[i][j] = min(dp[i][j], dp[i][k-1] + dp[k][j])
        
        return dp[0][n-1]