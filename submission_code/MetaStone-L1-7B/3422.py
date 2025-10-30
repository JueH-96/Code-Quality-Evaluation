class Solution:
    MOD = 10**9 + 7

    # Precompute factorial and inverse factorial up to 2000
    max_n = 2000
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % Solution.MOD

    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], Solution.MOD - 2, Solution.MOD)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % Solution.MOD

    def valueAfterKSeconds(self, n: int, k: int) -> int:
        a = n + k - 1
        b = n - 1
        return (fact[a] * inv_fact[b] % Solution.MOD) * inv_fact[a - b] % Solution.MOD