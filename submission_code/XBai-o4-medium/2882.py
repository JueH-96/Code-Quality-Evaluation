class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        values = []
        i = 1
        while True:
            current = pow(i, x)
            if current > n:
                break
            values.append(current)
            i += 1
        
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to make sum 0 (using no numbers)
        
        for num in values:
            for sum_ in range(n, num - 1, -1):
                dp[sum_] = (dp[sum_] + dp[sum_ - num]) % MOD
        
        return dp[n] % MOD