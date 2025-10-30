import math
from typing import List

class Solution:
  def maximumStrength(self, nums: List[int], k: int) -> int:
    """
    Calculates the maximum strength of k disjoint subarrays using dynamic programming.

    The solution uses a DP approach with space optimization. Let dp[j][i] be the
    maximum strength from using j subarrays in the prefix nums[0...i-1].
    The state transition can be optimized from O(n) to O(1) by maintaining a
    running maximum of a part of the recurrence relation.
    
    The space is optimized from O(n*k) to O(n) by realizing that the computation
    for j subarrays only depends on the results for j-1 subarrays.

    Time complexity: O(n * k)
    Space complexity: O(n)
    """
    n = len(nums)
    
    # ps[i] will store the sum of the first i elements of nums
    ps = [0] * (n + 1)
    for i in range(n):
        ps[i+1] = ps[i] + nums[i]

    neg_inf = -math.inf

    # prev_dp[i] stores max strength for (j-1) subarrays in nums[0...i-1]
    # For j=1, j-1=0, so prev_dp is initialized for 0 subarrays, where strength is always 0.
    prev_dp = [0] * (n + 1) 

    for j in range(1, k + 1):
        # curr_dp[i] will store max strength for j subarrays in nums[0...i-1]
        curr_dp = [neg_inf] * (n + 1)
        
        # running_max_term is the optimization term:
        # max_{j-1 <= p < i} (prev_dp[p] - multiplier * ps[p])
        running_max_term = neg_inf 
        
        # multiplier = (-1)^(j+1) * (k - j + 1)
        multiplier = (k - j + 1) * (1 if j % 2 != 0 else -1)
        
        # To form j non-empty subarrays, we need at least i >= j elements
        for i in range(j, n + 1):
            # Update running_max_term with the term for p = i - 1.
            running_max_term = max(running_max_term, prev_dp[i-1] - multiplier * ps[i-1])
            
            # max strength with j subarrays, with the j-th one ending at index i-1
            strength_ending_at_i = running_max_term + multiplier * ps[i]
            
            # curr_dp[i] is the max strength with j subarrays in nums[:i].
            # It's either the max from nums[:i-1] (curr_dp[i-1]), or the max
            # with the last subarray ending at i-1 (strength_ending_at_i).
            curr_dp[i] = max(curr_dp[i-1], strength_ending_at_i)

        # Update dp array for the next iteration (j+1)
        prev_dp = curr_dp
        
    return int(prev_dp[n])