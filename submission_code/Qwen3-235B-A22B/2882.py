class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to make sum 0 (using no numbers)
        
        j = 1
        while True:
            # Compute j^x
            current = 1
            for _ in range(x):
                current *= j
                if current > n:
                    break
            if current > n:
                break
            v = current
            # Update the dp array in reverse to avoid using the same number multiple times
            for s in range(n, v - 1, -1):
                dp[s] = (dp[s] + dp[s - v]) % MOD
            j += 1
        
        return dp[n] % MOD