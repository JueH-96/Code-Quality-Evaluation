class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        mod = 10**9 + 7
        total = m * n
        
        # Precompute factorials and inverse factorials up to 'total'
        fact = [1] * (total + 1)
        inv_fact = [1] * (total + 1)
        for i in range(1, total + 1):
            fact[i] = fact[i-1] * i % mod
        inv_fact[total] = pow(fact[total], mod - 2, mod)
        for i in range(total - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % mod
        
        # For every pair of cells, if chosen, they appear in exactly C(total-2, k-2) arrangements.
        # So, the total contribution is:
        #   Answer = C(total-2, k-2) * (sum of Manhattan distances over all cell pairs)
        # Compute binom = C(total-2, k-2)
        binom = fact[total - 2] * inv_fact[k - 2] % mod * inv_fact[total - k] % mod
        
        # Now, compute the sum of Manhattan distances between every pair of cells in the m x n grid.
        # Note the Manhattan distance between cells (r1, c1) and (r2, c2) is |r1-r2| + |c1-c2|.
        # We can separate the contribution into two parts: rows and columns.
        #
        # For the rows:
        #   Each row r appears n times.
        #   The contribution from rows is: n^2 * sum_{0 <= r1 < r2 < m} (r2 - r1).
        # And the closed form for S_row = sum_{0 <= r1 < r2 < m} (r2 - r1) is m*(m-1)*(m+1)//6.
        #
        # For the columns:
        #   Each column c appears m times.
        #   The contribution from columns is: m^2 * sum_{0 <= c1 < c2 < n} (c2 - c1),
        #   with the closed form S_col = n*(n-1)*(n+1)//6.
        
        # Compute modular multiplicative inverse of 6 modulo mod
        inv6 = pow(6, mod - 2, mod)
        
        # Row contribution:
        row_part = (n * n) % mod
        row_diff = (m * (m - 1)) % mod
        row_diff = (row_diff * (m + 1)) % mod
        row_diff = (row_diff * inv6) % mod
        row_contribution = (row_part * row_diff) % mod

        # Column contribution:
        col_part = (m * m) % mod
        col_diff = (n * (n - 1)) % mod
        col_diff = (col_diff * (n + 1)) % mod
        col_diff = (col_diff * inv6) % mod
        col_contribution = (col_part * col_diff) % mod

        total_manhattan = (row_contribution + col_contribution) % mod
        
        # Multiply by the number of ways to choose the other k-2 cells from the remaining total-2 cells.
        answer = binom * total_manhattan % mod
        return answer

# For running basic tests:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.distanceSum(2, 2, 2))  # Expected output: 8
    # Example 2:
    print(sol.distanceSum(1, 4, 3))  # Expected output: 20