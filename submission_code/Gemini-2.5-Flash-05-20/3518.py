import math
from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        N = len(b)
        K = 4 # The fixed size of array 'a'

        # prev_dp[j] will store the maximum score for the previous 'k' (a[k-1])
        # where b[j] was chosen for a[k-1].
        # Initialize with negative infinity to correctly handle cases where no valid path
        # exists to a certain state, and to ensure that max() works correctly even with
        # negative scores.
        prev_dp = [-math.inf] * N

        # Base case: k = 0 (for a[0])
        # For a[0], we need to choose b[j] such that there are K-1 (i.e., 3) elements
        # remaining after index j for a[1], a[2], and a[3].
        # So, the maximum index j for a[0] is N - 1 - (K - 1) = N - K.
        # This means j ranges from 0 to N - K.
        # The loop iterates up to N - (K-1) exclusive, which means j goes up to N - K.
        for j in range(N - (K - 1)):
            prev_dp[j] = a[0] * b[j]

        # Iterate for k from 1 to K-1 (i.e., for a[1], a[2], a[3])
        # k_idx represents the current 0-indexed position in 'a'
        for k_idx in range(1, K):
            current_dp = [-math.inf] * N
            
            # max_val_from_prev_k stores the maximum score found so far from the
            # previous DP layer (k_idx-1) considering indices strictly less than the
            # current 'j'.
            max_val_from_prev_k = -math.inf

            # Loop through possible indices 'j' for b[j] for the current a[k_idx].
            # The minimum index 'j' must be k_idx (since i_0 < i_1 < ... < i_k_idx).
            # The maximum index 'j' must leave enough elements in 'b' for the remaining
            # (K - 1 - k_idx) elements of 'a'.
            # So, j can go up to N - 1 - (K - 1 - k_idx) = N - (K - k_idx).
            # The range function's end is exclusive, so N - (K - k_idx).
            for j in range(k_idx, N - (K - 1 - k_idx)):
                # Update max_val_from_prev_k by taking the maximum of its current value
                # and the score from the previous layer at index (j-1).
                # This correctly finds the max of dp[k_idx-1][p] for all p < j.
                max_val_from_prev_k = max(max_val_from_prev_k, prev_dp[j-1])
                
                # If max_val_from_prev_k is still -inf, it means no valid path existed
                # to any index p < j in the previous layer. In this case, current_dp[j]
                # should remain -inf, indicating an invalid path.
                if max_val_from_prev_k != -math.inf:
                    current_dp[j] = a[k_idx] * b[j] + max_val_from_prev_k
            
            # After computing all current_dp values for the current k_idx,
            # update prev_dp to current_dp for the next iteration.
            prev_dp = current_dp

        # The final answer is the maximum value in the last computed DP array (prev_dp),
        # which contains the maximum scores for a[3] ending at various indices.
        return max(prev_dp)