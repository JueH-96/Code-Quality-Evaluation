class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        mod = 10**9 + 7

        if k > n - 1:
            return 0

        # Precompute factorial and inverse factorial
        fact = [1] * n
        for i in range(1, n):
            fact[i] = fact[i - 1] * i % mod

        inv_fact = [1] * n
        inv_fact[n - 1] = pow(fact[n - 1], mod - 2, mod)
        for i in range(n - 2, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod

        # Compute C(n-1, k)
        comb = (fact[n - 1] * inv_fact[k] % mod) * inv_fact[n - 1 - k] % mod

        # Compute (m -1)^(n -1 -k) mod mod
        pow_val = pow(m - 1, n - 1 - k, mod)

        # Compute final answer
        answer = m * comb % mod
        answer = answer * pow_val % mod
        return answer