class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Collect all i^x such that i^x <= n
        powers = []
        i = 1
        while True:
            p = i**x
            if p > n:
                break
            powers.append(p)
            i += 1
        
        # Let m = len(powers). We use a DP table of size (m+1) x (n+1).
        # dp[k][s] = number of ways to form sum s using the first k powers (unique usage).
        m = len(powers)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1  # base case: one way to form sum 0 by using none of the powers
        
        for k in range(1, m + 1):
            p = powers[k - 1]
            for s in range(n + 1):
                # We can always skip the k-th power, so start with ways from dp[k-1][s]
                dp[k][s] = dp[k - 1][s]
                # If we include this power (p), add ways from dp[k-1][s - p], if s >= p
                if s >= p:
                    dp[k][s] = (dp[k][s] + dp[k - 1][s - p]) % MOD
        
        return dp[m][n] % MOD