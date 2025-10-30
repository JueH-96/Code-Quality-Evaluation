class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        candidates = []
        i = 1
        while True:
            current = i ** x
            if current > n:
                break
            candidates.append(current)
            i += 1
        
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to form sum 0 (using no elements)
        
        for v in candidates:
            for s in range(n, v - 1, -1):
                dp[s] = (dp[s] + dp[s - v]) % MOD
        
        return dp[n] % MOD