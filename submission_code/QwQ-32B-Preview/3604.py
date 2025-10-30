class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Compute Stirling numbers of the second kind S(n, k)
        prev = [0] * (x + 1)
        prev[0] = 1
        for i in range(1, n + 1):
            curr = [0] * (x + 1)
            for k in range(1, x + 1):
                curr[k] = (k * prev[k] + prev[k - 1]) % MOD
            prev = curr
        
        # Compute factorial up to x for P(x, k)
        fact = [1] * (x + 1)
        for i in range(1, x + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        # Compute y^k for k from 1 to x
        yk = [1] * (x + 1)
        for k in range(1, x + 1):
            yk[k] = (yk[k - 1] * y) % MOD
        
        # Calculate total number of ways
        total = 0
        for k in range(1, x + 1):
            P = fact[x] * pow(fact[x - k], MOD - 2, MOD) % MOD if x - k >= 0 else 0
            term = prev[k] * P % MOD * yk[k] % MOD
            total = (total + term) % MOD
        
        return total