class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        t = n - k
        if t < 1:
            return 0
        
        a = n - 1
        b = t - 1
        
        # Precompute factorial and inverse factorial up to a
        fact = [1] * (a + 1)
        for i in range(1, a + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (a + 1)
        inv_fact[a] = pow(fact[a], MOD - 2, MOD)
        for i in range(a - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Compute combination C(a, b)
        if b < 0 or b > a:
            return 0
        comb = fact[a] * inv_fact[b] % MOD
        comb = comb * inv_fact[a - b] % MOD
        
        # Compute (m-1)^(t-1)
        pow_term = pow(m - 1, b, MOD)
        
        # Calculate the result
        result = m * comb % MOD
        result = result * pow_term % MOD
        return result