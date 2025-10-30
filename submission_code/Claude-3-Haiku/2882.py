class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Helper function to calculate the xth power of a number
        def power(num):
            return num ** x
        
        # Dynamic programming approach
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                if power(j) <= i:
                    dp[i] = (dp[i] + dp[i - power(j)]) % MOD
                else:
                    break
        
        return dp[n]