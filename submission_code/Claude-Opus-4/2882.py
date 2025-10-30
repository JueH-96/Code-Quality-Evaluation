class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i] = number of ways to form sum i
        dp = [0] * (n + 1)
        dp[0] = 1  # one way to form 0: use no numbers
        
        # Find the maximum number we need to consider
        # We need num^x <= n, so num <= n^(1/x)
        max_num = int(n ** (1/x))
        
        # For each number from 1 to max_num
        for num in range(1, max_num + 1):
            power = num ** x
            
            # If this power is greater than n, we can't use it
            if power > n:
                break
            
            # Update dp array from right to left
            # This ensures we don't use the same number multiple times
            for i in range(n, power - 1, -1):
                dp[i] = (dp[i] + dp[i - power]) % MOD
        
        return dp[n]