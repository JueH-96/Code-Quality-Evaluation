class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        # Generate all i^x values that are <= n
        nums = []
        i = 1
        while True:
            val = i ** x
            if val > n:
                break
            nums.append(val)
            i += 1
        
        # Initialize DP array
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to make sum 0 (use no elements)
        
        for num in nums:
            # Update dp in reverse to avoid using the same element multiple times
            for j in range(n, num - 1, -1):
                dp[j] = (dp[j] + dp[j - num]) % MOD
        
        return dp[n] % MOD