class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 1000000007
        max_size = n + k
        fact = [0] * max_size
        fact[0] = 1
        for i in range(1, max_size):
            fact[i] = (fact[i - 1] * i) % MOD
        inv_fact_k = pow(fact[k], MOD - 2, MOD)
        inv_fact_n_minus_1 = pow(fact[n - 1], MOD - 2, MOD)
        binom = (fact[n + k - 1] * inv_fact_k % MOD * inv_fact_n_minus_1 % MOD) % MOD
        return binom