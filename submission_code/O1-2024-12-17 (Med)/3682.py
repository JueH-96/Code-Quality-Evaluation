class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7

        # Edge case: if n == 1, then k must be 0 to have any valid array
        if n == 1:
            return m % MOD if k == 0 else 0

        # Precompute factorials and inverse factorials for binomial coefficients
        # up to n (which is at most 10^5).
        fact = [1] * (n + 1)
        inv_fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        # Fermat's little theorem for modular inverse: inv_fact[n] = fact[n]^(MOD-2) mod MOD
        inv_fact[n] = pow(fact[n], MOD - 2, MOD)
        for i in reversed(range(n)):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

        def binom(n, r):
            if r < 0 or r > n:
                return 0
            return fact[n] * inv_fact[r] % MOD * inv_fact[n - r] % MOD

        # Number of ways to select which (n-k) segments => C(n-1, k)
        ways_partition = binom(n - 1, k)

        # Number of ways to choose the colors of (n-k) segments with adjacent differing
        # For the first segment we have m choices,
        # and for each of the remaining (n-k-1) segments we have (m - 1) choices.
        # (This is valid because consecutive segments must have different colors.)
        ways_color = m % MOD
        if n - k - 1 > 0:
            ways_color = (ways_color * pow(m - 1, n - k - 1, MOD)) % MOD

        # Final answer: multiply both parts
        return (ways_partition * ways_color) % MOD