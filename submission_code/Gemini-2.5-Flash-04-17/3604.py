class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7

        # Compute Stirling numbers of the second kind S(i, j)
        # S(i, j) is the number of ways to partition a set of i elements into j non-empty subsets.
        # dp_S[i][j] stores S(i, j) mod MOD
        # We need S(n, k) for k up to min(n, x).
        # The DP table size is (n+1) x (x+1), which is sufficient.
        dp_S = [[0 for _ in range(x + 1)] for _ in range(n + 1)]

        # Base case: S(0, 0) = 1 (partition an empty set into 0 non-empty subsets)
        dp_S[0][0] = 1

        # Compute DP table using the recurrence: S(i, j) = S(i-1, j-1) + j * S(i-1, j)
        # i represents the number of performers (from 1 to n)
        # j represents the number of non-empty stages (from 1 to x)
        for i in range(1, n + 1):
            # Iterate through possible number of non-empty stages j
            # S(i, j) is non-zero only if 1 <= j <= i.
            # We also only need S(i, j) for j <= x in the final summation.
            # So we iterate j from 1 up to min(i, x).
            for j in range(1, min(i, x) + 1):
                 # S(i-1, j-1): Performer i forms a new band (on a new stage).
                 #               The first i-1 performers formed j-1 bands.
                 # j * S(i-1, j): Performer i joins one of the existing j bands.
                 #                The first i-1 performers formed j bands.
                 dp_S[i][j] = (dp_S[i-1][j-1] + j * dp_S[i-1][j]) % MOD

        # The total number of ways is the sum over the number of non-empty stages k.
        # For a fixed number of non-empty stages k (1 <= k <= min(n, x)):
        # 1. Number of ways to assign n distinct performers to x distinct stages such that exactly k stages are non-empty.
        #    This is S(n, k) * P(x, k), where P(x, k) = x! / (x-k)! is the number of permutations of choosing k stages out of x.
        #    S(n, k) partitions n performers into k unlabeled non-empty groups. P(x, k) assigns these k groups to k distinct chosen stages.
        # 2. Given k non-empty stages (bands), each band can be awarded a score from 1 to y.
        #    There are y^k ways to assign scores to these k bands.
        # Total ways = Sum_{k=1}^{min(n, x)} [ (S(n, k) * P(x, k)) * y^k ] mod MOD

        total_ways = 0
        P_xk = 1  # Represents P(x, k) = x * (x-1) * ... * (x-k+1)
        pow_y_k = 1 # Represents y^k

        # Iterate through the possible number of non-empty stages, k
        # k must be at least 1 (since n >= 1) and at most min(n, x).
        for k in range(1, min(n, x) + 1):
            # Calculate P(x, k) iteratively: P(x, k) = P(x, k-1) * (x - (k-1))
            # P(x, 0) = 1.
            # For k=1, P(x, 1) = P(x, 0) * (x - 0) = 1 * x = x.
            # The current P_xk holds P(x, k-1) from the previous iteration (or 1 for k=1).
            # The factor for step k is (x - (k-1)).
            # Since k <= min(n, x) and x >= 1, if k >= 1, then k-1 >= 0.
            # If k <= x, then k-1 < x, so x - (k-1) > 0.
            # The term (x - (k - 1)) is always positive for k in the loop range.
            P_xk = (P_xk * (x - (k - 1))) % MOD

            # Calculate y^k iteratively: y^k = y^(k-1) * y
            # y^0 = 1.
            # The current pow_y_k holds y^(k-1) from the previous iteration (or 1 for k=1).
            pow_y_k = (pow_y_k * y) % MOD

            # Get S(n, k) from the DP table
            S_nk = dp_S[n][k]

            # Calculate the term for this k: S(n, k) * P(x, k) * y^k mod MOD
            term = (S_nk * P_xk) % MOD
            term = (term * pow_y_k) % MOD

            # Add the term to the total ways
            total_ways = (total_ways + term) % MOD

        return total_ways