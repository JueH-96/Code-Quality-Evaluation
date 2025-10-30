import math
from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        """
        Finds the maximum sum of k non-overlapping subarrays of nums, 
        where each subarray has a length of at least m.
        Uses dynamic programming with space optimization.

        The problem asks for the maximum sum of k non-overlapping subarrays, each of length at least m.
        Let dp[i][j] be the maximum sum of j subarrays using the first i elements of nums.

        We can define two possibilities for dp[i][j]:
        1. The j-th subarray does not end at index i-1. In this case, dp[i][j] = dp[i-1][j].
        2. The j-th subarray ends exactly at index i-1. Let this subarray have length l >= m. 
           It starts at index i-l. The sum of this subarray is prefix_sum[i] - prefix_sum[i-l].
           The previous j-1 subarrays must use elements up to index i-l-1.
           The maximum sum for this case is max_{l>=m} (dp[i-l][j-1] + prefix_sum[i] - prefix_sum[i-l]).
           Let f[i][j] denote this maximum sum when the j-th subarray ends at i-1.
           f[i][j] = prefix_sum[i] + max_{l>=m} (dp[i-l][j-1] - prefix_sum[i-l])
           Let p = i-l. Since l >= m, p <= i-m. Also l <= i, so p >= 0.
           f[i][j] = prefix_sum[i] + max_{0 <= p <= i-m} (dp[p][j-1] - prefix_sum[p])

        The recurrence relation is: dp[i][j] = max(dp[i-1][j], f[i][j])

        To compute f[i][j] efficiently, we can maintain the maximum value of (dp[p][j-1] - prefix_sum[p])
        as i increases. Let max_term[idx][j] = max_{0 <= p <= idx} (dp[p][j-1] - prefix_sum[p]).
        Then f[i][j] = prefix_sum[i] + max_term[i-m][j].

        We can optimize the space complexity from O(n*k) to O(n) by storing only the DP values
        for the current number of subarrays (j) and the previous number (j-1).

        Args:
            nums: The input list of integers.
            k: The number of non-overlapping subarrays.
            m: The minimum length of each subarray.

        Returns:
            The maximum sum achievable.
        """
        n = len(nums)
        
        # Calculate prefix sums
        # prefix_sum[i] stores the sum of nums[0...i-1]
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
            
        # Initialize DP arrays for space optimization
        # dp_prev[i] stores the maximum sum of j-1 subarrays using the first i elements
        # dp_curr[i] stores the maximum sum of j subarrays using the first i elements
        
        # Initialize dp_prev for j=0 (0 subarrays). Max sum is 0 for any prefix length.
        dp_prev = [0] * (n + 1)  

        # Use negative infinity to represent impossible states
        neg_inf = -float('inf')

        # Iterate through the number of subarrays from 1 to k
        for j in range(1, k + 1):
            # Initialize dp_curr for the current number of subarrays j
            # dp_curr[i] will store the max sum using j subarrays using the first i elements.
            # Initialize with -infinity. dp_curr[0] remains -inf for j > 0.
            dp_curr = [neg_inf] * (n + 1) 
            
            # max_prev_dp_minus_prefix stores max(dp_prev[q] - prefix_sum[q]) for q <= i - m
            # This is needed to efficiently calculate f[i][j].
            max_prev_dp_minus_prefix = neg_inf
            
            # Iterate through the length of the prefix considered.
            # i represents using the first i elements (indices 0 to i-1).
            for i in range(1, n + 1):
                
                # Calculate f[i][j]: max sum achievable if the j-th subarray ends exactly at index i-1.
                # This term corresponds to the case where the last element nums[i-1] is included in the j-th subarray.
                f_i_j = neg_inf 
                
                # A subarray ending at i-1 must have length at least m, so i must be at least m.
                if i >= m:
                    # We need to find max_{0 <= q <= i-m} (dp_prev[q] - prefix_sum[q]).
                    # q = i - m is the relevant index for updating the running maximum.
                    q = i - m 
                    
                    # Calculate the term (dp_prev[q] - prefix_sum[q]) for the current q.
                    term_q = neg_inf
                    # Check if dp_prev[q] is valid (not -inf) before subtraction.
                    if dp_prev[q] != neg_inf:
                        term_q = dp_prev[q] - prefix_sum[q]
                    
                    # Update the maximum of these terms encountered so far for indices up to q = i - m.
                    max_prev_dp_minus_prefix = max(max_prev_dp_minus_prefix, term_q)

                    # If a valid maximum exists (meaning it's possible to form j-1 subarrays
                    # ending before or at index i-m-1), calculate f[i][j].
                    if max_prev_dp_minus_prefix != neg_inf:
                        # f[i][j] = prefix_sum[i] + max_{0 <= q <= i-m} (dp_prev[q] - prefix_sum[q])
                        f_i_j = prefix_sum[i] + max_prev_dp_minus_prefix
                
                # Apply the main DP recurrence: dp[i][j] = max(dp[i-1][j], f[i][j])
                # dp_curr[i] = max(dp_curr[i-1], f_i_j)
                # dp_curr[i-1] represents the max sum using j subarrays with the first i-1 elements (case 1).
                # f_i_j represents the max sum using j subarrays where the j-th subarray ends at i-1 (case 2).
                dp_curr[i] = max(dp_curr[i-1], f_i_j)

            # After iterating through all i for the current j, update dp_prev to dp_curr 
            # for the next iteration (j+1). Use list() to ensure a copy is made.
            dp_prev = list(dp_curr) 

        # The final answer is the maximum sum using k subarrays considering all n elements.
        # This is stored in dp_prev[n] after the loop finishes (dp_prev holds the result for j=k).
        result = dp_prev[n]
        
        # The constraints guarantee k <= floor(n / m), implying a solution always exists,
        # though the sum might be negative.
        return result