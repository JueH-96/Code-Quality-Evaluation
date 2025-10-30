class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        # Generate all possible candidates
        candidates = []
        m = 1
        while m ** x <= n:
            candidates.append(m)
            m += 1
        
        # Initialize DP array
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to make sum 0 (using no elements)
        
        # Process each candidate
        for m in candidates:
            val = m ** x
            # Update DP in reverse to avoid using the same candidate multiple times
            for s in range(n, val - 1, -1):
                dp[s] = (dp[s] + dp[s - val]) % MOD
        
        return dp[n] % MOD