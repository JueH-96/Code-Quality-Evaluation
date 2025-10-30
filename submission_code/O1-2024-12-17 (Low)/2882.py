class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute all possible x-th powers up to n
        powers = []
        base = 1
        while True:
            val = base**x
            if val > n:
                break
            powers.append(val)
            base += 1
        
        # Let m be the count of precomputed powers
        m = len(powers)
        
        # dp[s][i] will store the number of ways to reach sum s
        # using the first i powers (i.e. powers[0], ..., powers[i-1])
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Base case: there's exactly 1 way to make sum 0 with any i (by choosing nothing).
        for i in range(m + 1):
            dp[0][i] = 1
        
        # Fill the DP table
        for i in range(1, m + 1):
            power_val = powers[i - 1]  # i-th power in the 1-based indexing of 'i'
            for s in range(1, n + 1):
                # Ways to form s without using the current power
                dp[s][i] = dp[s][i - 1]
                
                # If we can include the current power
                if s >= power_val:
                    dp[s][i] += dp[s - power_val][i - 1]
                
                dp[s][i] %= MOD
        
        return dp[n][m] % MOD