class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7

        def comb(n, r):
            if r > n:
                return 0
            if r == 0 or r == n:
                return 1
            r = min(r, n - r)
            numer = 1
            denom = 1
            for i in range(r):
                numer = numer * (n - i) % MOD
                denom = denom * (i + 1) % MOD
            return numer * pow(denom, MOD - 2, MOD) % MOD

        def sum_of_distances(size):
            total = 0
            for i in range(size):
                total += i * (size - i - 1)
            return total

        total_cells = m * n
        total_distance = 0

        # Calculate the sum of distances for rows
        row_distance = sum_of_distances(m)
        col_distance = sum_of_distances(n)

        # Calculate the number of ways to choose k pieces from total_cells
        total_ways = comb(total_cells, k)

        # Calculate the contribution of row distances
        row_contribution = row_distance * n % MOD * comb(total_cells - 2, k - 2) % MOD
        total_distance = (total_distance + row_contribution) % MOD

        # Calculate the contribution of column distances
        col_contribution = col_distance * m % MOD * comb(total_cells - 2, k - 2) % MOD
        total_distance = (total_distance + col_contribution) % MOD

        return total_distance