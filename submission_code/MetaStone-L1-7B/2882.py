class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        # Generate all xth powers <= n
        power_list = []
        a = 1
        while True:
            current = a ** x
            if current > n:
                break
            power_list.append(current)
            a += 1
        
        # Sort the power list to ensure correct processing order
        power_list.sort()
        
        # Initialize DP array
        mod = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to make sum 0
        
        for m in power_list:
            for s in range(n, m - 1, -1):
                dp[s] = (dp[s] + dp[s - m]) % mod
        
        return dp[n]