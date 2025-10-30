class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Find max number whose x-th power is <= n
        max_num = int(pow(n, 1/x))
        
        # dp[i][j] represents number of ways to make sum i using numbers up to j
        dp = [[0] * (max_num + 1) for _ in range(n + 1)]
        
        # Base case - empty sum can be made in 1 way
        for j in range(max_num + 1):
            dp[0][j] = 1
            
        # For each number from 1 to max_num
        for j in range(1, max_num + 1):
            power = pow(j, x)
            # For each target sum from 1 to n
            for i in range(n + 1):
                # Don't use current number
                dp[i][j] = dp[i][j-1]
                # Use current number if possible
                if i >= power:
                    dp[i][j] = (dp[i][j] + dp[i-power][j-1]) % MOD
                    
        return dp[n][max_num]