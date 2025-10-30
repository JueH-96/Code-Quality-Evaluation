class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        """
        We can show that after k seconds, the last element of the array
        a[n-1] equals the binomial coefficient C(n-1 + k, n-1) (or equivalently C(n-1 + k, k)).
        We'll compute this modulo 10^9+7 using precomputed factorials and
        modular inverses (Fermat's little theorem).
        """
        
        MOD = 10**9 + 7
        
        # We only need factorials up to (n + k)
        max_val = n + k
        
        # Precompute factorials and their inverses
        fact = [1] * (max_val + 1)
        for i in range(2, max_val + 1):
            fact[i] = fact[i-1] * i % MOD
        
        # Fermat's little theorem for modular inverse: a^(MOD-2) ≡ 1/a (mod MOD)
        inv_fact = [1] * (max_val + 1)
        inv_fact[max_val] = pow(fact[max_val], MOD - 2, MOD)
        
        # Fill in inverses in descending order
        for i in range(max_val - 1, 0, -1):
            inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD
        
        # Function to compute nCk under modulo
        def nCk(n_, k_):
            if k_ < 0 or k_ > n_:
                return 0
            return (fact[n_] * inv_fact[k_] % MOD) * inv_fact[n_ - k_] % MOD
        
        # We want C((n-1) + k, n-1) mod 10^9+7
        return nCk(n - 1 + k, n - 1) % MOD