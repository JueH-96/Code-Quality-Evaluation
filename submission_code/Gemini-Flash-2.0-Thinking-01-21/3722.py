from typing import List
import math

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # P[i] = sum of nums[0...i-1]
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + nums[i]
        
        # dp[i][j] = max sum using nums[0...i] ending with the j-th subarray at index i.
        dp = [[float('-inf')] * (k + 1) for _ in range(n)]
        
        # max_so_far[i][j] = max sum using nums[0...i] with j subarrays, last one ending at or before i.
        max_so_far = [[float('-inf')] * (k + 1) for _ in range(n)]
        
        # Conceptually, max_so_far_ext[-1][0] = 0.
        # This represents the maximum sum using -1 elements (empty prefix) with 0 subarrays is 0.
        # max_so_far_ext[i][0] = 0 for i >= -1.
        # max_so_far_ext[i][j] = float('-inf') for j > 0, i < j*m - 1.
        # max_so_far_ext[i][j] = max_so_far[i][j] for i >= 0.

        for j in range(1, k + 1):
            # max_prev_term_val stores max_{p_start_iter} (max_so_far_ext[p_start_iter-1][j-1] - P[p_start_iter])
            # where p_start_iter is a potential start index in nums for the j-th subarray ending at current index i.
            # The range for p_start_iter is [(j-1)*m, i - m + 1].
            # We update this running max as i increases.
            
            max_prev_term_val = float('-inf')

            # The current index i must be large enough to accommodate j subarrays of length at least m, ending at i.
            # Minimum i = j * m - 1.
            for i in range(j * m - 1, n):
                # The potential start index `p_start = i - m + 1` is now the largest possible start index
                # for a subarray of length >= m ending at `i`.
                # This `p_start` value is newly included in the range [(j-1)*m, i - m + 1] as `i` increases.
                
                p_start = i - m + 1 # The new potential start index in nums for the j-th subarray ending at i
                prev_end_idx = p_start - 1 # The end index in nums of the (j-1)-th subarray block
                
                prev_max_sum = float('-inf')
                
                # Get the max sum for j-1 subarrays ending at or before prev_end_idx.
                # This is max_so_far_ext[prev_end_idx][j-1].
                if j - 1 == 0: # Previous stage was 0 subarrays
                    # max_so_far_ext[prev_end_idx][0] is 0 if prev_end_idx >= -1.
                    # Since i >= j*m - 1 = m - 1 (for j=1), prev_end_idx = i - m >= -1.
                    # So prev_end_idx is always >= -1 here when j=1.
                    prev_max_sum = 0
                else: # Previous stage was j-1 > 0 subarrays
                    # max_so_far_ext[prev_end_idx][j-1] corresponds to max_so_far[prev_end_idx][j-1] if prev_end_idx >= 0.
                    # If j > 1, minimum i is j*m - 1. Minimum prev_end_idx = j*m - 1 - m = (j-1)m - 1.
                    # If m >= 1, (j-1)m - 1 >= 1*1 - 1 = 0. So prev_end_idx is always >= 0 if j > 1.
                    if prev_end_idx >= 0: # This is always true for j > 1 given the loop bounds for i.
                         prev_max_sum = max_so_far[prev_end_idx][j-1]
                    # else: prev_end_idx < 0, max_so_far_ext is float('-inf') for j-1 > 0. It's already initialized to float('-inf').

                # Update the running max `max_prev_term_val` with the term corresponding to the new p_start = i - m + 1.
                # The term is `max_so_far_ext[p_start-1][j-1] - P[p_start]`.
                if prev_max_sum != float('-inf'): # Check if the previous state was possible
                    term = prev_max_sum - P[p_start]
                    max_prev_term_val = max(max_prev_term_val, term)

                # Calculate dp[i][j].
                # dp[i][j] = sum(nums[p_start...i]) + max_so_far_ext[p_start-1][j-1]  (maximized over p_start)
                # dp[i][j] = (P[i+1] - P[p_start]) + max_so_far_ext[p_start-1][j-1]  (maximized over p_start)
                # dp[i][j] = P[i+1] + max_{p_start} (max_so_far_ext[p_start-1][j-1] - P[p_start])
                # The max_{p_start} is precisely `max_prev_term_val`.
                if max_prev_term_val != float('-inf'):
                    dp[i][j] = P[i+1] + max_prev_term_val

                # Update max_so_far[i][j].
                # max_so_far[i][j] = max sum using nums[0...i] with j subarrays ending AT OR BEFORE i.
                # It's max of max sum ending at i-1 with j subarrays (max_so_far[i-1][j])
                # and max sum ending exactly at i with j subarrays (dp[i][j]).
                
                max_so_far_i_minus_1_j = float('-inf')
                if i > 0:
                    max_so_far_i_minus_1_j = max_so_far[i-1][j]
                # If i is the first possible index for this j (i == j*m - 1), then i-1 is j*m - 2.
                # max_so_far[j*m - 2][j] is float('-inf') by definition for j > 0.
                # So the max(max_so_far[i-1][j], dp[i][j]) logic works correctly even for the first index.

                max_so_far[i][j] = max(max_so_far_i_minus_1_j, dp[i][j])

        # The final answer is max_so_far[n-1][k].
        # The problem guarantees that a solution exists (k*m <= n).
        # So the result should not be float('-inf').
        # If it somehow ends up being float('-inf'), it implies no valid configuration was found,
        # which contradicts the constraint. In a contest setting, we might return 0 or handle it.
        # Here, we can assume max_so_far[n-1][k] will be a valid number.

        return max_so_far[n-1][k]