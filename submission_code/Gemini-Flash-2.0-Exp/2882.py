class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 1
        
        power_values = []
        i = 1
        while i**x <= n:
            power_values.append(i**x)
            i += 1
        
        for power in power_values:
            for i in range(n, power - 1, -1):
                dp[i] = (dp[i] + dp[i - power]) % (10**9 + 7)
        
        return dp[n]