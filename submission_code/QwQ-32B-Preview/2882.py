class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Find all k where k^x <= n
        k_powers = []
        k = 1
        while True:
            k_pow = k ** x
            if k_pow > n:
                break
            k_powers.append(k_pow)
            k += 1
        
        # Initialize DP array
        dp = [0] * (n + 1)
        dp[0] = 1  # One way to make sum 0
        
        # Update DP array for each k^x
        for power in k_powers:
            for j in range(n, power - 1, -1):
                dp[j] = (dp[j] + dp[j - power]) % MOD
        
        return dp[n]