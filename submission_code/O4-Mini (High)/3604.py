class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        mod = 10**9 + 7
        
        # Compute Stirling numbers of the second kind S(n,k) for k=0..n
        # dp[k] will become S(i,k) after i iterations; initialize for i=0
        dp = [0] * (n + 1)
        dp[0] = 1
        
        # Recurrence: S(i, j) = S(i-1, j-1) + j * S(i-1, j)
        for i in range(1, n + 1):
            # update in reverse to avoid overwriting
            for j in range(i, 0, -1):
                dp[j] = (dp[j - 1] + j * dp[j]) % mod
            dp[0] = 0
        
        # Now dp[k] = S(n, k)
        # We want sum_{k=1..min(n,x)} S(n,k) * P(x,k) * y^k
        # where P(x,k) = x * (x-1) * ... * (x-k+1)
        maxk = min(n, x)
        ans = 0
        p = 1       # will hold P(x,k)
        pow_y = 1   # will hold y^k
        
        for k in range(1, maxk + 1):
            # Update P(x, k) = P(x, k-1) * (x - k + 1)
            p = p * (x - k + 1) % mod
            # Update y^k
            pow_y = pow_y * y % mod
            # Add contribution for exactly k non-empty stages
            ans = (ans + dp[k] * p % mod * pow_y) % mod
        
        return ans