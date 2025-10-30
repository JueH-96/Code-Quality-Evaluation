class Solution:
    MOD = 10**9 + 7
    max_n = 1000

    # Precompute Stirling numbers of the second kind
    stirling = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    stirling[0][0] = 1
    for i in range(1, max_n + 1):
        for j in range(1, i + 1):
            stirling[i][j] = (j * stirling[i-1][j] + stirling[i-1][j-1]) % MOD

    # Precompute factorials and inverse factorials modulo MOD
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD

    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

    def numberOfWays(self, n: int, x: int, y: int) -> int:
        res = 0
        max_k = min(x, n)
        for k in range(1, max_k + 1):
            if x < k:
                continue
            # Calculate permutations of x taken k at a time
            permute = fact[x] * inv_fact[x - k] % MOD
            # Stirling number for partitioning n into k non-empty subsets
            s = stirling[n][k]
            # y^k modulo MOD
            y_pow = pow(y, k, MOD)
            # Compute the term for current k and add to the result
            term = permute * s % MOD
            term = term * y_pow % MOD
            res = (res + term) % MOD
        return res