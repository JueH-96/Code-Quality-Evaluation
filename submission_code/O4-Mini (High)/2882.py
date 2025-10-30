class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        # Precompute all k^x <= n
        powers = []
        k = 1
        while True:
            p = k**x
            if p > n:
                break
            powers.append(p)
            k += 1
        
        # dp[s] = number of ways to form sum s using each power at most once
        dp = [0] * (n + 1)
        dp[0] = 1
        
        # 0/1 knapsack over the powers array
        for p in powers:
            for s in range(n, p - 1, -1):
                dp[s] = (dp[s] + dp[s - p]) % MOD
        
        return dp[n]