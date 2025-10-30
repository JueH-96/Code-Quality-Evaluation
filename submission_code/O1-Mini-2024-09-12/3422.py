class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        m = n + k - 1
        fact = [1] * (m + 1)
        for i in range(1, m + 1):
            fact[i] = fact[i-1] * i % MOD
        inv_fact = [1] * (m + 1)
        inv_fact[m] = pow(fact[m], MOD - 2, MOD)
        for i in range(m - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i + 1) % MOD
        return fact[m] * inv_fact[k] % MOD * inv_fact[n - 1] % MOD