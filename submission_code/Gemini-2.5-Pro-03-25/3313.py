import math
from typing import List

# Using float('-inf') for negative infinity represents impossible states or extremely small values.
neg_inf = float('-inf')

class Solution:
    """
    Finds the maximum strength achievable by selecting k disjoint subarrays from nums.
    The strength is defined by the formula:
    strength = sum[1] * k - sum[2] * (k-1) + sum[3] * (k-2) - ... + sum[k] * 1
    where sum[i] is the sum of elements in the i-th selected subarray.
    k is guaranteed to be a positive odd integer.
    """
    def maximumStrength(self, nums: List[int], k: int) -> int:
        """
        Calculates maximum strength using dynamic programming with space optimization.
        
        The DP state is defined as:
        dp[i][j]: Maximum strength using the first i elements of nums (prefix nums[0...i-1]) 
                  selecting exactly j disjoint subarrays.
        
        The recurrence relation involves an optimization technique to avoid O(n^2 * k) complexity.
        Let C_j = (-1)^(j+1) * (k - j + 1).
        The recurrence relation is:
        dp[i][j] = max(dp[i-1][j], max_prev[i-1][j-1] + C_j * P[i])
        where P[i] is the prefix sum nums[0] + ... + nums[i-1],
        and max_prev[i-1][j-1] = max_{0 <= p < i} (dp[p][j-1] - C_j * P[p]).
        
        The max_prev term can be computed incrementally:
        max_prev[i][j-1] = max(max_prev[i-1][j-1], dp[i][j-1] - C_j * P[i])

        Time Complexity: O(n * k) because we fill the DP table states iteratively.
        Space Complexity: O(k) because we use space optimization with only two rows for the DP table.
        
        Args:
          nums: A list of integers.
          k: The number of disjoint subarrays to select (positive odd integer).

        Returns:
          The maximum possible strength as an integer.
        """
        n = len(nums)
        
        # Calculate prefix sums
        # P[i] stores the sum of nums[0...i-1]. P[0] = 0.
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + nums[i]
            
        # Initialize DP table with two rows for space optimization.
        # dp[curr/prev][j] stores max strength for index i/i-1 with j subarrays.
        # Row indices 0 and 1 alternate based on i % 2.
        dp = [[neg_inf] * (k + 1) for _ in range(2)]
        
        # Base case: With 0 subarrays, the strength is 0.
        # This applies regardless of how many elements are considered (i.e., for both rows).
        dp[0][0] = 0
        dp[1][0] = 0
        
        # Iterate through the number of subarrays from 1 to k
        for j in range(1, k + 1):
            # Calculate the coefficient C_j = (-1)^(j+1) * (k - j + 1)
            coeff = (k - j + 1)
            # Check parity of j for sign of C_j
            if j % 2 == 0: # j is even, (-1)^(j+1) is -1
                C_j = -coeff
            else: # j is odd, (-1)^(j+1) is 1
                C_j = coeff

            # `prev_max` tracks the value max_{0 <= p < i} (dp[p][j-1] - C_j * P[p]) across iterations of i.
            # This variable effectively stores max_prev[i-1][j-1] needed for dp[i][j] calculation.
            # Initialize `prev_max` for the state before considering the first element (i=0).
            # This corresponds to max_prev[0][j-1].
            # If j=1, dp[0][0]=0, P[0]=0, so max_prev[0][0] = dp[0][0] - C_1*P[0] = 0.
            # If j>1, dp[0][j-1]=-inf, so max_prev[0][j-1] = -inf.
            prev_max = 0 if j == 1 else neg_inf

            # Iterate through the elements of nums array (indices from 0 to n-1)
            # DP state index `i` corresponds to prefix nums[0...i-1]
            for i in range(1, n + 1):
                # Determine current and previous row indices based on i for the two DP rows.
                curr = i % 2
                prev = (i - 1) % 2
                 
                # Calculate helper term `dp[i][j-1] - C_j * P[i]`.
                # dp[i][j-1] is the max strength using prefix nums[0...i-1] with j-1 subarrays.
                # This value is available at dp[curr][j-1], computed in the previous iteration of outer loop j for the same i.
                term2 = neg_inf
                # Check if dp[curr][j-1] is a valid (reachable) state before using it.
                if dp[curr][j-1] != neg_inf:
                    term2 = dp[curr][j-1] - C_j * P[i]
                
                # Compute max_prev[i][j-1] using max_prev[i-1][j-1] (which is stored in `prev_max`)
                # and the just computed `term2`. This updates the running maximum needed for the DP transition optimization.
                # `current_max_prev_i` stores max_{0 <= p <= i} (dp[p][j-1] - C_j * P[p])
                current_max_prev_i = max(prev_max, term2)
                
                # Calculate the candidate strength if the j-th subarray ends at index i-1.
                # The formula is max_prev[i-1][j-1] + C_j * P[i].
                # `prev_max` holds the value of max_prev[i-1][j-1].
                candidate_strength_ending_here = neg_inf
                if prev_max != neg_inf:
                     candidate_strength_ending_here = prev_max + C_j * P[i]

                # Compute dp[i][j]: Max strength using prefix nums[0...i-1] with j subarrays.
                # It's the maximum of two cases:
                # 1. The j-th subarray ends before index i-1. Max strength is dp[i-1][j].
                # 2. The j-th subarray ends exactly at index i-1. Max strength is candidate_strength_ending_here.
                # dp[i-1][j] is stored in dp[prev][j].
                dp[curr][j] = max(dp[prev][j], candidate_strength_ending_here)

                # Update `prev_max` for the next iteration i+1. It should now hold max_{0 <= p <= i} ...
                # This value is `current_max_prev_i`.
                prev_max = current_max_prev_i
            
        # The final answer is the maximum strength using all n elements (prefix nums[0...n-1]) and k subarrays.
        # This result is stored in dp[n % 2][k].
        final_strength = dp[n % 2][k]
        
        # The problem constraints (k <= n) ensure it's always possible to select k subarrays,
        # e.g., by choosing k single-element subarrays. Thus, the result won't be negative infinity.
        # The result must be an integer as sums and coefficients are integers.
        return int(final_strength)