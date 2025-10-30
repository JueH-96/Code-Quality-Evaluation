class Solution:
    # Precompute factorials and inverse factorials modulo 10^9 + 7
    MOD = 10**9 + 7
    # MAX_N_GRID is the maximum possible value for m*n according to constraints
    MAX_N_GRID = 100000
    fact = [1] * (MAX_N_GRID + 1)
    invFact = [1] * (MAX_N_GRID + 1)
    # Flag to check if precomputation is done
    _precomputed = False

    @staticmethod
    def power(a, b, m):
        res = 1
        a %= m
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % m
            a = (a * a) % m
            b //= 2
        return res

    @staticmethod
    def modInverse(n, m):
        # Uses Fermat's Little Theorem for prime modulus
        # modInverse(n, m) returns a such that (n * a) % m == 1
        # Requires m to be prime and n != 0 mod m
        return Solution.power(n, m - 2, m)

    @staticmethod
    def precompute_factorials(max_val, m):
        if Solution._precomputed:
            return

        # Compute factorials
        for i in range(2, max_val + 1):
            Solution.fact[i] = (Solution.fact[i-1] * i) % m

        # Compute inverse factorials using modular inverse of the largest factorial
        if max_val >= 0:
             Solution.invFact[max_val] = Solution.modInverse(Solution.fact[max_val], m)
             # Compute inverse factorials for smaller values iteratively
             for i in range(max_val - 1, -1, -1):
                 # invFact[i] = invFact[i+1] * (i+1) % m
                 Solution.invFact[i] = (Solution.invFact[i+1] * (i + 1)) % m
        
        # Ensure base case 0! = 1, invFact[0] = 1
        # These are usually covered by the loops if max_val >= 0, but explicit ensures clarity.
        Solution.fact[0] = 1
        Solution.invFact[0] = 1
        
        Solution._precomputed = True


    @staticmethod
    def nCr_mod_p(n, r, m):
        if r < 0 or r > n:
            return 0
        # Ensure factorials are precomputed up to n
        # Based on constraints m*n <= 10^5 = MAX_N_GRID, nCr is called with n = m*n - 2,
        # which is within the precomputed range.
        if n > Solution.MAX_N_GRID:
             # This case should not happen based on constraints
             # If constraints change, this would need adjustment (e.g., dynamic calculation)
            raise ValueError(f"n={n} exceeds MAX_N_GRID={Solution.MAX_N_GRID} for precomputed factorials")

        # Precomputation is handled by the distanceSum method caller
        
        numerator = Solution.fact[n]
        denominator = (Solution.invFact[r] * Solution.invFact[n-r]) % m
        return (numerator * denominator) % m
        
    # Helper function to compute Sum_{0<=i,j<L} |i-j| mod MOD
    # This sum is equal to L * (L^2 - 1) / 3 for L >= 2.
    @staticmethod
    def sum_1d_abs_diff_mod_p(L, m):
        # Sum_{0<=i,j<L} |i-j|.
        # If L < 2, there are no pairs of distinct indices, so the sum is 0.
        if L < 2: return 0

        # The formula is L * (L^2 - 1) / 3
        L_mod = L % m
        L_sq = (L_mod * L_mod) % m
        # Calculate L_sq - 1, handle potential negative result from %
        L_sq_minus_1 = (L_sq - 1 + m) % m
        
        numerator = (L_mod * L_sq_minus_1) % m
        
        # Modular inverse of 3
        # Since MOD = 10^9 + 7 is prime and > 3, inv(3) exists.
        inv3 = Solution.modInverse(3, m)
        result = (numerator * inv3) % m
        return result


    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = self.MOD
        MAX_N_GRID = self.MAX_N_GRID
        
        # Ensure precomputation is done once for the necessary range
        # We need factorials up to m*n <= MAX_N_GRID.
        self.precompute_factorials(MAX_N_GRID, MOD)
        
        # Handle k < 2 case (no pairs between pieces)
        # Although constraints say k >= 2, this check is good practice.
        if k < 2:
            return 0

        N_total = m * n

        # Constraints: 2 <= m * n <= 10^5 and 2 <= k <= m * n
        # These guarantee N_total >= 2, k >= 2, and k <= N_total.
        # Thus, N_total - 2 >= 0, k - 2 >= 0, and k - 2 <= N_total - 2.
        # So C(N_total - 2, k - 2) is always well-defined with n >= r >= 0.

        # The total number of valid arrangements is C(N_total, k).
        # We are summing the total Manhattan distance over all these arrangements.
        # The sum of Manhattan distances for one arrangement (set S of k cells) is:
        # Sum_arr = Sum_{P1 in S, P2 in S, P1 != P2} MD(P1, P2) / 2 (to count each pair once)
        # The total sum over all arrangements is:
        # TotalSum = Sum_{all valid arrangements S} Sum_arr
        # We can swap the summation order:
        # TotalSum = (1/2) * Sum_{distinct cells P1, P2 in grid} Sum_{arrangements S containing P1, P2} MD(P1, P2)
        # The number of arrangements containing two specific distinct cells P1 and P2 is C(N_total - 2, k - 2).
        # This is constant for any pair of distinct cells P1, P2, given k >= 2 and N_total >= 2.
        # TotalSum = (1/2) * C(N_total - 2, k - 2) * Sum_{distinct cells P1, P2 in grid} MD(P1, P2)
        # The sum of Manhattan distances over all distinct pairs of cells in the grid is:
        # SumPairsDistance = Sum_{distinct cells P1, P2} (|r1 - r2| + |c1 - c2|) / 2 (for unordered pairs)
        # Sum_{distinct cells P1, P2} MD(P1, P2) = Sum_{ordered distinct cells P1, P2} MD(P1, P2)
        # Sum_{ordered distinct cells P1, P2} MD(P1, P2) = Sum_{ordered distinct P1, P2} |r1 - r2| + Sum_{ordered distinct P1, P2} |c1 - c2|
        # Sum_{ordered distinct pairs ((r_a, c_a), (r_b, c_b))} |r_a - r_b|
        # = Sum_{all ordered pairs ((r_a, c_a), (r_b, c_b))} |r_a - r_b| (since terms with r_a=r_b are 0)
        # = Sum_{0<=r_a<m, 0<=r_b<m} Sum_{0<=c_a<n, 0<=c_b<n} |r_a - r_b|
        # = Sum_{0<=r_a<m, 0<=r_b<m} n * n * |r_a - r_b|
        # = n^2 * Sum_{0<=r_a<m, 0<=r_b<m} |r_a - r_b|
        # We know Sum_{0<=i,j<L} |i-j| = L(L^2-1)/3.
        # So Sum_{ordered distinct pairs} |r_a - r_b| = n^2 * m(m^2-1)/3.
        # Similarly, Sum_{ordered distinct pairs} |c_a - c_b| = m^2 * n(n^2-1)/3.

        # Sum of Manhattan distances over ordered distinct pairs = n^2 * m(m^2-1)/3 + m^2 * n(n^2-1)/3 mod MOD

        M = m % MOD
        N_val = n % MOD # Using N_val for n

        # Calculate Sum_{0<=i,j<L} |i-j| mod MOD for L=m and L=n
        sum_row_abs_diff = self.sum_1d_abs_diff_mod_p(m, MOD)
        sum_col_abs_diff = self.sum_1d_abs_diff_mod_p(n, MOD)

        # Calculate n^2 mod MOD and m^2 mod MOD
        n_sq = (N_val * N_val) % MOD
        m_sq = (M * M) % MOD

        # Sum of |r_a - r_b| over ordered distinct pairs of cells = n^2 * sum_row_abs_diff
        row_sum_ordered_dist = (n_sq * sum_row_abs_diff) % MOD

        # Sum of |c_a - c_b| over ordered distinct pairs of cells = m^2 * sum_col_abs_diff
        col_sum_ordered_dist = (m_sq * sum_col_abs_diff) % MOD

        # Sum of Manhattan distances over ordered distinct pairs
        total_sum_ordered_dist = (row_sum_ordered_dist + col_sum_ordered_dist) % MOD

        # Sum of Manhattan distances over unordered distinct pairs = total_sum_ordered_dist / 2
        # This is the SumPairsDistance term in the derivation.
        inv2 = self.modInverse(2, MOD)
        sum_pairs_dist = (total_sum_ordered_dist * inv2) % MOD

        # TotalSum = C(N_total - 2, k - 2) * SumPairsDistance mod MOD
        total_sum = (num_combinations * sum_pairs_dist) % MOD

        return total_sum