import sys
from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        # dp0: max cost ending with current element having coeff +1
        # dp1: max cost ending with current element having coeff -1

        # Base case i = 0
        # nums[0] must be the first element of the first subarray, so its coefficient is +1.
        # The maximum cost for the prefix containing only nums[0] is nums[0] * (+1).
        dp0 = nums[0]
        # It is impossible for nums[0] to have a coefficient of -1 as it is the first element.
        # Initialize with a value lower than any possible minimum cost.
        # Minimum cost can be around 10^5 * (-10^9) = -10^14.
        dp1 = -sys.maxsize # A sufficiently small number

        # Iterate from the second element
        for i in range(1, n):
            # Store the previous DP states before calculating the current ones
            prev_dp0 = dp0
            prev_dp1 = dp1

            # Calculate the max cost for prefix nums[0..i] ending with nums[i] having coeff +1 (next_dp0).
            # This can happen in two ways:
            # 1. Split before nums[i]. nums[i] starts a new subarray, so its coeff is +1.
            #    The cost for the prefix nums[0..i-1] was the maximum possible, max(prev_dp0, prev_dp1).
            #    Total cost = max(prev_dp0, prev_dp1) + nums[i].
            # 2. Do not split before nums[i]. nums[i] continues the subarray from nums[i-1].
            #    For nums[i] to have coeff +1, nums[i-1] must have had coeff -1 (state prev_dp1).
            #    Adding nums[i] to this sequence (without splitting) results in adding +nums[i]
            #    to the total accumulated cost (due to the sign flip from -1 to +1).
            #    Total cost = prev_dp1 + nums[i].
            # next_dp0 is the maximum of these two possibilities.
            next_dp0 = max(max(prev_dp0, prev_dp1) + nums[i], prev_dp1 + nums[i])

            # Calculate the max cost for prefix nums[0..i] ending with nums[i] having coeff -1 (next_dp1).
            # This can only happen by not splitting before nums[i].
            # If nums[i] has coeff -1, nums[i-1] must have had coeff +1 (state prev_dp0).
            # Adding nums[i] to this sequence (without splitting) results in adding -nums[i]
            # to the total accumulated cost (due to the sign flip from +1 to -1).
            # Total cost = prev_dp0 - nums[i].
            next_dp1 = prev_dp0 - nums[i]

            # Update the DP states for the current index i
            dp0 = next_dp0
            dp1 = next_dp1

        # After iterating through all elements up to n-1, the maximum total cost
        # for the entire array nums[0..n-1] is the maximum of the costs ending
        # with nums[n-1] having either coefficient +1 or -1.
        return max(dp0, dp1)