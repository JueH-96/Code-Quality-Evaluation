import math
from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # For n=1, there's only one element and no possible splits.
        # The cost is cost(0,0) = nums[0] * (-1)^(0-0) = nums[0].
        if n == 1:
            return nums[0]

        # prev_max_cost_plus: max total cost ending at previous index (i-1),
        # where nums[i-1] contributed +nums[i-1].
        # prev_max_cost_minus: max total cost ending at previous index (i-1),
        # where nums[i-1] contributed -nums[i-1].

        # Base case for i = 0:
        # nums[0] is always the start of the first subarray.
        # Its contribution is always +nums[0].
        prev_max_cost_plus = nums[0]
        prev_max_cost_minus = -math.inf  # Represents an impossible state

        # Iterate from the second element (index 1) to the end of the array
        for i in range(1, n):
            # Calculate current_max_cost_plus for nums[i]:
            # nums[i] contributes +nums[i].
            # This can happen in two ways:
            # 1. nums[i] starts a new subarray. The total cost up to i-1 is max(prev_max_cost_plus, prev_max_cost_minus).
            # 2. nums[i] continues the previous subarray. For nums[i] to be +nums[i], nums[i-1] must have been -nums[i-1].
            #    So, it comes from prev_max_cost_minus + nums[i].
            # The max operation naturally combines these:
            current_max_cost_plus = max(prev_max_cost_plus, prev_max_cost_minus) + nums[i]

            # Calculate current_max_cost_minus for nums[i]:
            # nums[i] contributes -nums[i].
            # This can only happen if nums[i] continues the previous subarray.
            # For nums[i] to be -nums[i], nums[i-1] must have been +nums[i-1].
            # So, it comes from prev_max_cost_plus - nums[i].
            current_max_cost_minus = prev_max_cost_plus - nums[i]

            # Update prev states for the next iteration
            prev_max_cost_plus = current_max_cost_plus
            prev_max_cost_minus = current_max_cost_minus
        
        # The overall maximum total cost is the maximum of the two states at the end of the array.
        return max(prev_max_cost_plus, prev_max_cost_minus)