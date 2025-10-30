class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] will be the number of good arrays of length i with exactly j pairs of consecutive equal elements
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # Base case: an array of length 1 can't have any consecutive pairs
        dp[1][0] = m
        
        # Fill the dp table
        for i in range(2, n + 1):
            for j in range(k + 1):
                # Case 1: arr[i-1] != arr[i-2], contribute from dp[i-1][j]
                dp[i][j] = dp[i-1][j] * (m - 1) % MOD
                
                # Case 2: arr[i-1] == arr[i-2], contribute from dp[i-1][j-1] if j > 0
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i-1][j-1]) % MOD
        
        return dp[n][k]