class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] stores the value of a[j] after i seconds
        dp = [[0] * n for _ in range(k + 1)]
        
        # Initialize the array with 1s at time 0
        for j in range(n):
            dp[0][j] = 1
        
        # Iterate through each second
        for i in range(1, k + 1):
            # Iterate through each element in the array
            for j in range(n):
                # Calculate the sum of all preceding elements plus the element itself
                sum_val = 0
                for l in range(j + 1):
                    sum_val = (sum_val + dp[i - 1][l]) % MOD
                dp[i][j] = sum_val
        
        # Return the value of a[n - 1] after k seconds
        return dp[k][n - 1]