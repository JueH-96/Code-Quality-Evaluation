MOD = 10**9 + 7

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        # Edge case for n = 1
        if n == 1:
            return m if k == 0 else 0
        
        # Precompute factorials and inverse factorials up to n-1
        N = n  # we need up to n-1
        fact = [1] * (N + 1)
        inv_fact = [1] * (N + 1)
        for i in range(1, N + 1):
            fact[i] = fact[i-1] * i % MOD
        
        # Fermat's little theorem for modular inverse
        inv_fact[N] = pow(fact[N], MOD-2, MOD)
        for i in range(N, 0, -1):
            inv_fact[i-1] = inv_fact[i] * i % MOD
        
        # Compute C(n-1, k)
        if k > n-1:
            return 0
        c = fact[n-1] * inv_fact[k] % MOD * inv_fact[n-1-k] % MOD
        
        # Number of ways to pick the segment values:
        # There are (n-k) segments, first has m choices,
        # each subsequent segment has (m-1) choices:
        ways_segments = m * pow(m-1, n-k-1, MOD) % MOD
        
        return c * ways_segments % MOD