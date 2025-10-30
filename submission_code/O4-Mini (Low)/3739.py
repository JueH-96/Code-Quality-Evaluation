class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7

        # Fast exponentiation
        def modpow(a, e):
            res = 1
            a %= MOD
            while e:
                if e & 1:
                    res = res * a % MOD
                a = a * a % MOD
                e >>= 1
            return res

        # Precompute factorials and inverse factorials up to N = m*n
        N = m * n
        fac = [1] * (N + 1)
        for i in range(1, N + 1):
            fac[i] = fac[i-1] * i % MOD
        invfac = [1] * (N + 1)
        invfac[N] = modpow(fac[N], MOD - 2)
        for i in range(N, 0, -1):
            invfac[i-1] = invfac[i] * i % MOD

        # Binomial coefficient
        def comb(n, r):
            if r < 0 or r > n:
                return 0
            return fac[n] * invfac[r] % MOD * invfac[n-r] % MOD

        # If k < 2, no pairs => sum = 0
        if k < 2:
            return 0

        # Sum over all unordered pairs of Manhattan distances on the grid
        # Use formula: sum_{i<j} |i-j| over rows times n^2 plus similarly over columns
        inv6 = modpow(6, MOD - 2)

        # sum_{d=1..m-1} d*(m-d) = m*(m-1)*(m+1)/6
        sum_row = m * (m-1) % MOD * (m+1) % MOD * inv6 % MOD
        # each row-pair contributes for every pair of columns => n^2
        part_x = sum_row * (n * n % MOD) % MOD

        sum_col = n * (n-1) % MOD * (n+1) % MOD * inv6 % MOD
        part_y = sum_col * (m * m % MOD) % MOD

        total_dist_sum = (part_x + part_y) % MOD

        # Each unordered pair of pieces is counted in C(N-2, k-2) arrangements
        ways = comb(N-2, k-2)
        return total_dist_sum * ways % MOD