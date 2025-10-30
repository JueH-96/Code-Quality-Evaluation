from typing import List
from collections import defaultdict

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        # We need to choose a non-empty subsequence such that its alternating sum equals k.
        # The alternating sum is computed using the order of selection.
        # We want to maximize the product subject to it not exceeding 'limit'.
        # We can use dynamic programming where we iterate over the numbers (in their original order)
        # and decide whether to take each number to update our subsequence.
        #
        # In our DP state, we will track:
        #  - current alternating sum, "diff"
        #  - the number of elements chosen so far (count).
        #  - the best (maximum) product achieved for that state.
        #
        # When adding an element x, its effect on the alternating sum depends on its index
        # in the subsequence. If the current count is even, then the new element goes to an even index;
        # its value is added. If the count is odd, then x is subtracted.
        #
        # We also only consider transitions in which the product does not exceed limit.
        #
        # The alternating sum from any subsequence can lie in a limited range.
        # The maximum possible sum in absolute value occurs when all chosen numbers are 12,
        # and at most 150 numbers may be chosen, so |diff| <= 150 * 12 = 1800.
        # We do not create a full DP table over all these indices, but using a dictionary we
        # key by (diff, parity, count) where count is the number of selected elements.
        # We need count mainly to decide the sign of the next added element.
        
        # Initialize DP state:
        # dp[(diff, parity, count)] = best product achieving that state.
        # We start with an "empty" state.
        dp = {}
        dp[(0, 0, 0)] = 1  # diff=0, count=0 (so next index would be 0 (even)), product=1
        
        for x in nums:
            new_dp = dict(dp)  # copy current state and then update with new choices
            for (diff, parity, cnt), prod in dp.items():
                new_prod = prod * x
                if new_prod > limit:
                    continue
                # The next index in the subsequence is cnt (0-indexed).
                # If cnt is even, then x goes into an even index and is added;
                # if odd then x is subtracted.
                if cnt % 2 == 0:
                    new_diff = diff + x
                else:
                    new_diff = diff - x
                new_state = (new_diff, (cnt+1) % 2, cnt+1)
                # Update with maximum product for that state
                if new_state in new_dp:
                    new_dp[new_state] = max(new_dp[new_state], new_prod)
                else:
                    new_dp[new_state] = new_prod
            dp = new_dp
        
        ans = -1
        # Finally, among all non-empty states (cnt>=1) with alternating sum equal to k
        # we want the maximum product which is not greater than limit.
        for (diff, parity, cnt), prod in dp.items():
            if cnt > 0 and diff == k:
                ans = max(ans, prod)
        return ans