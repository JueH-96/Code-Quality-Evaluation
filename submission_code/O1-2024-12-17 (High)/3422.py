class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        # We want to compute the binomial coefficient C(n-1+k, k),
        # since after k seconds, a[n-1] = binomial(n-1 + k, k).

        # Precompute factorials up to n+k
        max_val = n + k
        fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = (fact[i - 1] * i) % MOD

        # Precompute modular inverses of the factorials (invFact)
        invFact = [1] * (max_val + 1)
        invFact[max_val] = pow(fact[max_val], MOD - 2, MOD)  # Fermat's little theorem
        for i in range(max_val, 0, -1):
            invFact[i - 1] = (invFact[i] * i) % MOD

        # Function to compute binomial coefficient C(x, y) % MOD
        def binom(x, y):
            return (fact[x] * (invFact[y] * invFact[x - y] % MOD)) % MOD

        # The answer is C(n - 1 + k, k) % MOD
        return binom(n - 1 + k, k)