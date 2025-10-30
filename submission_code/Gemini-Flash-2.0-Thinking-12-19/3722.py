from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)

        # Use a large negative number to represent negative infinity for integer sums.
        # The maximum possible sum is bounded (k * N * max(abs(nums_i))).
        # The minimum possible sum for a valid solution is also bounded (k * m * min(nums_i)).
        # -10^15 is sufficiently small to represent impossible states.
        NEGATIVE_INF = -10**15 

        # prefix_sum[i] = sum of first i elements (nums[0]...nums[i-1])
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]

        # prev_max_so_far[i] stores max_so_far for j-1 blocks using first i elements (indices 0..i-1)
        # curr_max_so_far[i] stores max_so_far for j blocks using first i elements (indices 0..i-1)
        # curr_dp[i] stores max sum ending at index i-1 using j blocks
        
        # Initialize for j=0: max sum of 0 blocks is 0, regardless of elements used
        prev_max_so_far = [0] * (n + 1) 

        # Initialize templates for j >= 1 states (impossible states are -inf)
        # max_so_far[i][j] and dp[i][j] are -inf if i < j*m
        curr_max_so_far_template = [NEGATIVE_INF] * (n + 1)
        curr_dp_template = [NEGATIVE_INF] * (n + 1)

        # Start DP for j = 1 to k
        for j in range(1, k + 1):
            # Reset curr_max_so_far and curr_dp for the current j iteration
            curr_max_so_far = curr_max_so_far_template[:]
            curr_dp = curr_dp_template[:]
            
            # max_prev_term tracks max_{ (j-1)*m <= p <= current_i-m } (prev_max_so_far[p] - prefix_sum[p])
            max_prev_term = NEGATIVE_INF 

            # Iterate through possible end indices i-1 (represented by i from 1 to n)
            # A state with j blocks requires at least j*m elements.
            # The loop for i should start from j*m.
            # States for i < j*m will remain at their initial NEGATIVE_INF value.
            for i in range(j * m, n + 1):
                # Update max_prev_term.
                # The candidate index for the previous state (using j-1 blocks) is p_candidate = i - m.
                # This index p_candidate represents the number of elements used by the previous j-1 blocks.
                # The current j-th block starts at index p_candidate and ends at i-1. Its length is i - p_candidate = m.
                # The condition (j-1)*m <= p <= i-m comes from:
                # 1. Current block length >= m: i - p >= m => p <= i - m.
                # 2. Previous j-1 blocks need >= (j-1)*m elements: p >= (j-1)*m.
                # So the p index for prev_max_so_far is in the range [(j-1)*m, i-m].
                # As i increases, i-m increases, expanding the upper bound of p.
                # The new p value considered at step i is exactly i-m.
                
                p_candidate = i - m
                
                # p_candidate is guaranteed to be >= (j-1)*m because i >= j*m.
                # p_candidate is guaranteed to be >= 0 when i >= j*m and m >= 1.
                # Example: j=1, m=1, i>=1. p = i-1 >= 0.
                # Example: j=2, m=1, i>=2. p = i-1 >= 1.
                # Example: j=1, m=2, i>=2. p = i-2 >= 0.
                # Example: j=2, m=2, i>=4. p = i-2 >= 2.
                # Example: j=1, m=3, i>=3. p = i-3 >= 0.
                # Example: j=2, m=3, i>=6. p = i-3 >= 3.
                # The minimum value of i-m when i starts at j*m is j*m - m = (j-1)*m. Since j>=1, m>=1, (j-1)*m >= 0.
                # So p_candidate is always a valid non-negative index.

                # Check if the previous state prev_max_so_far[p_candidate] is achievable (not NEGATIVE_INF)
                if prev_max_so_far[p_candidate] != NEGATIVE_INF:
                   # The term to maximize over p is prev_max_so_far[p] - prefix_sum[p].
                   # When considering ending the current j-th block at index i-1, starting at p,
                   # the sum is prev_max_so_far[p] + sum(nums[p...i-1])
                   # = prev_max_so_far[p] + prefix_sum[i] - prefix_sum[p]
                   # = prefix_sum[i] + (prev_max_so_far[p] - prefix_sum[p])
                   # We maximize this over p. prefix_sum[i] is constant for fixed i.
                   # So we maximize (prev_max_so_far[p] - prefix_sum[p]).
                   # The new candidate p is i-m.
                   max_prev_term = max(max_prev_term, prev_max_so_far[p_candidate] - prefix_sum[p_candidate])

                # Calculate curr_dp[i] (max sum ending at i-1 with j blocks)
                # This is prefix_sum[i] + max_{over relevant p} (prev_max_so_far[p] - prefix_sum[p])
                # The max over relevant p up to i-m is stored in max_prev_term.
                if max_prev_term != NEGATIVE_INF:
                   curr_dp[i] = prefix_sum[i] + max_prev_term
                else:
                   # If max_prev_term is still NEGATIVE_INF, it's impossible to form j blocks ending at i-1
                   curr_dp[i] = NEGATIVE_INF

                # Calculate curr_max_so_far[i] (max sum using first i elements with j blocks, ending anywhere)
                # Option 1: The j-th block does not end at i-1. Max sum is curr_max_so_far[i-1].
                # Option 2: The j-th block ends at i-1. Max sum is curr_dp[i].
                
                # curr_max_so_far[i-1] refers to the state using first i-1 elements with j blocks.
                # This value comes from the current j iteration, computed in the previous step i-1.
                # For the first step of the inner loop (i = j*m), curr_max_so_far[i-1] = curr_max_so_far[j*m - 1].
                # Since j*m - 1 < j*m, curr_max_so_far[j*m - 1] is NEGATIVE_INF from the initial template.
                # Thus, max(NEGATIVE_INF, curr_dp[j*m]) correctly results in curr_dp[j*m].
                # For i > j*m, curr_max_so_far[i-1] holds the correct value from the previous iteration of the inner loop.
                
                if i == j * m:
                    # For the first element where j blocks are possible, max_so_far is just the max sum ending there.
                    # The case where the j-th block doesn't end at i-1 isn't possible if i is the very first index where j blocks fit.
                    curr_max_so_far[i] = curr_dp[i]
                else:
                    curr_max_so_far[i] = max(curr_max_so_far[i-1], curr_dp[i])

            # After iterating through i for the current j, update prev_max_so_far for the next iteration of j
            # prev_max_so_far now holds the max_so_far values for j blocks,
            # to be used as the 'previous' values when calculating for j+1 blocks.
            prev_max_so_far[:] = curr_max_so_far[:]

        # The final answer is the max sum using k blocks from all N elements
        # This is stored in max_so_far[n][k], which is prev_max_so_far[n] after the last iteration.
        result = prev_max_so_far[n]

        # The problem constraints (1 <= k <= floor(n / m)) guarantee that it is possible to find
        # at least one configuration of k non-overlapping subarrays of length at least m.
        # Therefore, the maximum sum will be a finite number, not NEGATIVE_INF.
        # The result will always be >= k * m * min(nums), which is much larger than NEGATIVE_INF.
        
        return result