class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        max_nk = n + k - 1
        factorial = [1] * (max_nk + 1)
        for i in range(1, max_nk + 1):
            factorial[i] = factorial[i - 1] * i % MOD
        inv_fact = [1] * (max_nk + 1)
        inv_fact[max_nk] = pow(factorial[max_nk], MOD - 2, MOD)
        for i in range(max_nk - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        result = factorial[n + k - 1] * inv_fact[k] % MOD
        result = result * inv_fact[n - 1] % MOD
        return result