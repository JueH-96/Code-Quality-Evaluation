class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # We need to compute C(n-1 + k, k) mod 1e9+7
        # Maximum value for (n - 1 + k) is (1000 - 1 + 1000) = 1999, so we can
        # precompute factorials up to 2000 safely.
        
        max_val = n + k  # a small buffer to be safe
        
        # Precompute factorials and inverse factorials
        fact = [1] * (max_val + 1)
        invfact = [1] * (max_val + 1)
        
        for i in range(2, max_val + 1):
            fact[i] = fact[i-1] * i % MOD
        
        # Fermat's little theorem for modular inverse:
        # a^(MOD-2) mod MOD is the inverse of a mod MOD, when MOD is prime
        invfact[max_val] = pow(fact[max_val], MOD-2, MOD)
        for i in reversed(range(max_val)):
            invfact[i] = invfact[i+1] * (i+1) % MOD
        
        # Combination function nCk mod
        def comb(n_, k_):
            if k_ > n_ or k_ < 0:
                return 0
            return (fact[n_] * invfact[k_] % MOD) * invfact[n_ - k_] % MOD
        
        # After k updates, the (n-1)-th element is C((n-1)+k, k)
        return comb(n - 1 + k, k) % MOD