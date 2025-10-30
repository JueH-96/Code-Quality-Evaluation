class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = 10**9 + 7
        powers = []
        k = 1
        while True:
            power = k ** x
            if power > n:
                break
            powers.append(power)
            k += 1
        
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to make sum 0 (using no elements)
        
        for num in powers:
            for j in range(n, num - 1, -1):
                dp[j] = (dp[j] + dp[j - num]) % mod
        
        return dp[n]