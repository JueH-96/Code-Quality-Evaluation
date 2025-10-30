MOD = 10**9 + 7

class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        # We need to compute the value at a[n - 1] after k seconds.
        # Observation: Each second, every element a[i] becomes sum(a[0], ..., a[i]).
        # This means after k seconds, a[i] = binom(i + k, k) (since we are effectively applying k partial summations
        # to an array of ones).
        # We need a[n - 1] = binom((n - 1) + k, k) mod MOD.
        
        # Precompute factorials and inverse factorials up to (n - 1) + k.
        max_val = n - 1 + k
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)
        
        for i in range(1, max_val + 1):
            fact[i] = fact[i-1] * i % MOD
        
        # Fermat's little theorem for computing inverses mod MOD (MOD is prime)
        inv_fact[max_val] = pow(fact[max_val], MOD - 2, MOD)
        
        for i in range(max_val, 0, -1):
            inv_fact[i-1] = inv_fact[i] * i % MOD
        
        # binom(n-1+k, k)
        numerator = fact[n - 1 + k]
        denominator = (inv_fact[k] * inv_fact[n - 1]) % MOD
        result = numerator * denominator % MOD
        return result