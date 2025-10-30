import math
from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        """
        Calculates the maximum total cost by optimally splitting the array into subarrays.
        
        This method uses dynamic programming with space optimization. At each index `i`,
        we maintain two values:
        - dp_add: The maximum cost of the prefix nums[0...i] where nums[i] starts a new
                  subarray (and thus has a positive sign).
        - dp_sub: The maximum cost of the prefix nums[0...i] where nums[i] continues
                  a subarray (and thus has a negative sign).
        """
        n = len(nums)
        
        # dp_add: max cost for a prefix ending at i, where nums[i] has a positive sign.
        # dp_sub: max cost for a prefix ending at i, where nums[i] has a negative sign.
        
        # Base case for index 0: nums[0] must start the first subarray.
        dp_add = nums[0]
        # This state is impossible for index 0, so initialize to negative infinity.
        dp_sub = -math.inf

        for i in range(1, n):
            # Store values from the previous step (i-1) to compute the current step (i).
            prev_dp_add = dp_add
            prev_dp_sub = dp_sub

            # Case 1: nums[i] starts a new subarray (cost: +nums[i]).
            # This can follow any valid partition of nums[0...i-1].
            # We take the max cost up to i-1, which is max(prev_dp_add, prev_dp_sub).
            dp_add = max(prev_dp_add, prev_dp_sub) + nums[i]

            # Case 2: nums[i] continues a subarray (cost: -nums[i]).
            # This requires that nums[i-1] had a positive sign in its subarray context.
            dp_sub = prev_dp_add - nums[i]
            
        # The final answer is the maximum possible cost for the entire array.
        return max(dp_add, dp_sub)