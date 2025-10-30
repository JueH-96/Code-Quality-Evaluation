class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        min_k = min(n, x)
        
        # Precompute factorial and inverse factorial
        fact = [1] * (x + 1)
        for i in range(1, x + 1):
            fact[i] = fact[i-1] * i % MOD
        inv_fact = [1] * (x + 1)
        inv_fact[x] = pow(fact[x], MOD-2, MOD)
        for i in range(x-1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        # Precompute Stirling numbers S(n,k) for k from 0 to min_k
        prev = [0] * (min_k + 1)
        prev[0] = 1
        for i in range(1, n + 1):
            curr = [0] * (min_k + 1)
            for j in range(1, min_k + 1):
                curr[j] = (j * prev[j] + prev[j-1]) % MOD
            prev = curr
        
        # Calculate the total number of ways
        total = 0
        for k in range(1, min_k + 1):
            P = fact[x] * inv_fact[x - k] % MOD
            S = prev[k]
            yk = pow(y, k, MOD)
            total = (total + P * S % MOD * yk % MOD) % MOD
        return total