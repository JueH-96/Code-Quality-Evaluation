from typing import List
import math

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # dp[j] will store the maximum score using the first k elements of a,
        # with the k-th element taken from b[j].
        # Initialize with a sufficiently small number representing negative infinity.
        # Using float('-inf') is standard and safe.
        dp = [-math.inf] * n

        # k = 1 (using a[0])
        # dp[j] = a[0] * b[j] for j >= 0
        # This means we pick b[j] as the first element (i_0 = j).
        # This is valid for any j from 0 to n-1.
        for j in range(n):
            dp[j] = a[0] * b[j]

        # k = 2, 3, 4
        # For each step k, we compute dp[k][j] based on dp[k-1][i] where i < j.
        # We use O(N) space by keeping only the previous DP array (`prev_dp`)
        # and computing the current DP array (`dp`).
        for k in range(2, 5):
            prev_dp = dp # dp from previous stage (k-1)
            dp = [-math.inf] * n # dp for current stage (k), initialized to -inf

            # max_val_up_to_j_minus_1 will store the maximum of prev_dp values that
            # could precede the current index j.
            # Specifically, for computing dp[k][j], we need max(dp[k-1][i]) for i < j,
            # where i is a valid index for stage k-1.
            # The minimum valid index for dp[k-1] is (k-1) - 1 = k-2.
            # So we need max(prev_dp[i]) for k-2 <= i < j.
            
            max_val_up_to_j_minus_1 = -math.inf
            
            # The minimum index from which a valid previous DP value could come is k-2.
            min_prev_idx = k - 2

            # The current index j for stage k must be at least k-1.
            # We iterate j from 0 to n-1 to correctly update the running maximum
            # `max_val_up_to_j_minus_1` for each possible current index j.
            for j in range(n):
                # Update the running maximum `max_val_up_to_j_minus_1`.
                # When considering index `j` for the current stage (k), the previous element
                # must have been at an index `i` such that `i < j`.
                # If `j-1` is a valid index for the previous DP stage (k-1), meaning `j-1 >= min_prev_idx`,
                # then `prev_dp[j-1]` is a potential value to include in the maximum
                # of values *less than* `j`.
                # We update `max_val_up_to_j_minus_1` to include `prev_dp[j-1]`.
                # This running maximum will hold max(prev_dp[i]) for i in range(min_prev_idx, j).
                if j - 1 >= min_prev_idx:
                     # `max` correctly handles -inf, so this updates the max with reachable scores.
                     max_val_up_to_j_minus_1 = max(max_val_up_to_j_minus_1, prev_dp[j-1])
                
                # If the current index j is a valid position for the k-th element (i.e., j >= k-1)
                # And if a valid previous state was found (i.e., max_val_up_to_j_minus_1 is not -inf)
                # Then we can compute the score ending at index j for stage k.
                # The score is a[k-1] * b[j] + max_val_up_to_j_minus_1 (which is max score using k-1 elements ending before j).
                if j >= k - 1 and max_val_up_to_j_minus_1 is not -math.inf:
                    dp[j] = a[k - 1] * b[j] + max_val_up_to_j_minus_1

        # After the loop for k=4, dp[j] contains the maximum score using 4 elements (a[0]..a[3]),
        # where the 4th element (multiplied by a[3]) is b[j], for j >= 3.
        # The overall maximum score is the maximum value found anywhere in the final dp array.
        # Since indices < 3 in the final dp array were initialized to -inf and never
        # updated to a finite value (as j >= k-1 = 3 for k=4), `max(dp)` correctly
        # gives the maximum over relevant indices (j >= 3).
        return max(dp)