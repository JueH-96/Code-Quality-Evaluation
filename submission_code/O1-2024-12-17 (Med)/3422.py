class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        # We need to compute the binomial coefficient C(n + k - 1, n - 1) % MOD
        # for 1 <= n, k <= 1000. We'll precompute factorials and inverse factorials
        # up to (n + k), which is at most 2000.

        max_val = n + k  # upper bound on factorial index

        # Precompute factorials mod MOD
        fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = (fact[i - 1] * i) % MOD

        # Precompute inverse factorials mod MOD using Fermat's little theorem
        inv_fact = [1] * (max_val + 1)
        inv_fact[max_val] = pow(fact[max_val], MOD - 2, MOD)  # inverse of fact[max_val]
        for i in range(max_val - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

        def comb(a, b):
            if b < 0 or b > a:
                return 0
            return (fact[a] * inv_fact[b] % MOD) * inv_fact[a - b] % MOD

        return comb(n + k - 1, n - 1) % MOD