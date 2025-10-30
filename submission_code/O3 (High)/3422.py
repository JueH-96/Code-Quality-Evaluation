class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        """
        The value of the last element after k prefix–sum operations on an initial
        array of ones is the binomial coefficient C(n + k − 1, k).
        We compute it modulo 1_000_000_007.
        """
        MOD = 1_000_000_007

        # We need factorials up to n + k - 1
        N = n + k - 1
        fact = [1] * (N + 1)
        for i in range(2, N + 1):
            fact[i] = fact[i - 1] * i % MOD

        # Compute inverse factorials using Fermat's little theorem
        inv_fact = [1] * (N + 1)
        inv_fact[N] = pow(fact[N], MOD - 2, MOD)
        for i in range(N - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

        # C(N, k) = fact[N] * inv_fact[k] * inv_fact[N-k] (mod MOD)
        res = fact[N] * inv_fact[k] % MOD
        res = res * inv_fact[n - 1] % MOD
        return res