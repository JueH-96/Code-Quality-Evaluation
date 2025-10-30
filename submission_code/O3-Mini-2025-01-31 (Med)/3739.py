MOD = 10**9 + 7

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        # Total number of cells in the grid.
        T = m * n
        
        # Precompute factorials and inverse factorials up to T.
        # Since m*n <= 10^5, this is efficient.
        max_val = T
        fact = [1] * (max_val + 1)
        invfact = [1] * (max_val + 1)
        for i in range(2, max_val + 1):
            fact[i] = fact[i - 1] * i % MOD
        invfact[max_val] = pow(fact[max_val], MOD - 2, MOD)
        for i in range(max_val, 0, -1):
            invfact[i - 1] = invfact[i] * i % MOD

        # Function to compute combinations modulo MOD.
        def nCr(n_, r_):
            if r_ < 0 or r_ > n_:
                return 0
            return fact[n_] * invfact[r_] % MOD * invfact[n_ - r_] % MOD

        # Now, observe that for every valid arrangement, the sum of distances
        # between every pair of pieces can be decomposed as follows:
        # Instead of summing over placements, we sum over all pairs of cells.
        # For a pair of distinct cells (cell A and cell B) that are at Manhattan
        # distance 'd', they appear together in any arrangement that chooses (k-2)
        # pieces from the remaining T-2 cells.
        # Hence, the contribution of that specific pair is:
        #   d * C(T-2, k-2)
        #
        # The total contribution is then:
        #   C(T-2, k-2) * (Sum of Manhattan distances for all unordered pairs of cells)
        #
        # We can separate the Manhattan distance into a row part and a column part:
        #   |row1 - row2| and |col1 - col2|
        # And sum over rows and columns separately.
        
        # For rows:
        # There are n cells in each row. The contribution from any two distinct rows i and j is
        #   |i - j| * (n * n)
        # And the sum for rows becomes:
        #   n*n * sum_{1 <= i < j <= m} (j - i)
        # Note that:
        #   sum_{1 <= i < j <= m} (j - i) = sum_{d=1}^{m-1} d * (m - d)
        # This sum has a closed form: m*(m-1)*(m+1)/6.
        sum_rows = m * (m - 1) % MOD * (m + 1) % MOD
        sum_rows = sum_rows * pow(6, MOD - 2, MOD) % MOD
        
        # For columns:
        # There are m cells in each column. Similarly, the contribution is:
        #   m*m * sum_{1 <= i < j <= n} (j - i)
        # With closed form: n*(n-1)*(n+1)/6.
        sum_cols = n * (n - 1) % MOD * (n + 1) % MOD
        sum_cols = sum_cols * pow(6, MOD - 2, MOD) % MOD
        
        # Total Manhattan distance sum for all unordered cell pairs.
        total_manhattan = ((n * n % MOD) * sum_rows + (m * m % MOD) * sum_cols) % MOD
        
        # Factor representing the number of arrangements that include a given pair of cells.
        ways = nCr(T - 2, k - 2)
        answer = ways * total_manhattan % MOD
        return answer

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.distanceSum(2, 2, 2))  # Output: 8
    # Example 2:
    print(sol.distanceSum(1, 4, 3))  # Output: 20