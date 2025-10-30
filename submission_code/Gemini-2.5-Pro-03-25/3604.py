import sys
# Setting recursion depth is not strictly necessary for this iterative solution,
# but good practice if recursion was potentially involved.
# sys.setrecursionlimit(2000)

class Solution:
    """
    This class provides a solution to calculate the number of ways an event
    with performers, stages, and scores can take place.
    """
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        """
        Calculates the total number of possible ways the event can take place,
        modulo 10^9 + 7.

        An event consists of:
        1. Assigning 'n' performers to 'x' stages. Performers on the same stage
           form a band. Stages can remain empty.
        2. Awarding each band (formed on non-empty stages) a score from 1 to 'y'.

        Two events are considered different if either the performer assignments
        differ or the band scores differ.

        The total number of ways is calculated using the formula derived from
        combinatorial principles:
        Total Ways = Sum_{k=1 to min(n, x)} [P(x, k) * S(n, k) * y^k] % mod
        where:
        - k is the number of non-empty stages (bands).
        - P(x, k) = x! / (x-k)! is the number of k-permutations of x, representing
          the ways to choose 'k' stages out of 'x' and assign the 'k' bands
          to them distinctly.
        - S(n, k) is the Stirling number of the second kind, representing the
          number of ways to partition a set of 'n' performers into 'k'
          non-empty groups (bands).
        - y^k is the number of ways to assign scores (1 to y) independently
          to each of the 'k' bands.
        - mod = 10^9 + 7.

        Args:
            n: The number of performers (1 <= n <= 1000).
            x: The number of stages (1 <= x <= 1000).
            y: The maximum score for a band (1 <= y <= 1000).

        Returns:
            The total number of ways modulo 10^9 + 7.
        """
        mod = 10**9 + 7
        # The number of bands 'k' can range from 1 up to min(n, x),
        # as each band needs at least one performer (k <= n) and
        # each band needs a distinct stage (k <= x).
        m = min(n, x)

        # --- Step 1: Calculate Stirling numbers S(n, k) for k = 0 to m ---
        # We use dynamic programming with space optimization O(m).
        # dp[j] will store S(i, j) after the i-th outer iteration (processing i elements).
        # Initialize dp array for S(0, k).
        dp = [0] * (m + 1)
        dp[0] = 1  # Base case: S(0, 0) = 1 (one way to partition an empty set into 0 non-empty subsets)

        # Iterate through the number of elements 'i' from 1 to n.
        for i in range(1, n + 1):
            # Calculate S(i, j) using the recurrence relation:
            # S(i, j) = S(i-1, j-1) + j * S(i-1, j)
            # We iterate 'j' downwards from min(i, m) to 1. This ensures that when
            # calculating dp[j] (S(i, j)), we use dp[j-1] which holds S(i-1, j-1)
            # from the previous state (or computed in this iteration if j=1 uses dp[0])
            # and dp[j] which holds S(i-1, j) from the previous state.
            for j in range(min(i, m), 0, -1):
                # Update dp[j] to S(i, j)
                # dp[j-1] corresponds to S(i-1, j-1)
                # dp[j] corresponds to S(i-1, j) before the update
                dp[j] = (dp[j-1] + (j * dp[j]) % mod) % mod
            
            # After updating S(i, 1) to S(i, m), we set S(i, 0) = 0 for the next iteration.
            # dp[0] held S(i-1, 0) which was used to calculate dp[1] = S(i, 1).
            # Now, set dp[0] to represent S(i, 0) = 0 for i > 0.
            dp[0] = 0

        # After n iterations, dp[k] holds S(n, k) for k = 0 to m.
        stirling_n = dp

        # --- Step 2: Calculate the final sum using the formula ---
        total_ways = 0
        # current_perm will iteratively calculate P(x, k) = x * (x-1) * ... * (x-k+1)
        current_perm = 1  # Represents P(x, 0) = 1 initially
        # current_pow_y will iteratively calculate y^k
        current_pow_y = 1 # Represents y^0 = 1 initially

        # Iterate through the possible number of bands 'k' from 1 to m.
        for k in range(1, m + 1):
            # Update current_perm to P(x, k) using P(x, k) = P(x, k-1) * (x - k + 1)
            # Since k <= m <= x, the factor (x - k + 1) is always >= 1.
            current_perm = (current_perm * (x - k + 1)) % mod

            # Update current_pow_y to y^k using y^k = y^(k-1) * y
            current_pow_y = (current_pow_y * y) % mod

            # Retrieve the precomputed Stirling number S(n, k)
            stirling_nk = stirling_n[k]

            # Calculate the contribution for 'k' bands: P(x, k) * S(n, k) * y^k % mod
            # Multiply P(x, k) and S(n, k) modulo mod
            term_k = (current_perm * stirling_nk) % mod
            # Multiply the result by y^k modulo mod
            term_k = (term_k * current_pow_y) % mod

            # Add the contribution for this 'k' to the total sum, modulo mod
            total_ways = (total_ways + term_k) % mod

        # The final result is the accumulated sum modulo mod.
        return total_ways