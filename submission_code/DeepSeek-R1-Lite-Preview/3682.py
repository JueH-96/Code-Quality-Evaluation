class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        if n < 1 or k < 0 or k > n - 1:
            return 0
        # Precompute factorial and inverse factorial
        fact = [1] * n
        for i in range(1, n):
            fact[i] = fact[i - 1] * i % MOD
        inv_fact = [1] * n
        inv_fact[n - 1] = pow(fact[n - 1], MOD - 2, MOD)
        for i in range(n - 2, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        # Compute C(n-1, k)
        if k < 0 or k > n - 1:
            return 0
        C = fact[n - 1] * inv_fact[k] % MOD
        C = C * inv_fact[n - 1 - k] % MOD
        # Compute (m - 1)^(n - 1 - k)
        power = pow(m - 1, n - 1 - k, MOD)
        # Compute the final answer
        answer = m * C % MOD
        answer = answer * power % MOD
        return answer