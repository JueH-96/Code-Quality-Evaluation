class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10 ** 9 + 7
        max_nx = max(n, x) + 1

        # Precompute factorials and inverse factorials
        factorial = [1] * (max_nx)
        inv_factorial = [1] * (max_nx)
        for i in range(1, max_nx):
            factorial[i] = factorial[i - 1] * i % MOD
        inv_factorial[max_nx - 1] = pow(factorial[max_nx - 1], MOD - 2, MOD)
        for i in range(max_nx - 2, -1, -1):
            inv_factorial[i] = inv_factorial[i + 1] * (i + 1) % MOD

        # Function to compute binomial coefficients modulo MOD
        def comb(n, k):
            if k < 0 or k > n:
                return 0
            return factorial[n] * inv_factorial[k] % MOD * inv_factorial[n - k] % MOD

        # Precompute Stirling numbers of the second kind
        S = [[0] * (n + 1) for _ in range(n + 1)]
        S[0][0] = 1
        for i in range(1, n + 1):
            S[i][0] = 0
            for j in range(1, i + 1):
                S[i][j] = (j * S[i - 1][j] + S[i - 1][j - 1]) % MOD

        total_ways = 0
        for k in range(1, min(n, x) + 1):
            term = comb(x, k) * factorial[k] % MOD
            term = term * S[n][k] % MOD
            term = term * pow(y, k, MOD) % MOD
            total_ways = (total_ways + term) % MOD

        return total_ways