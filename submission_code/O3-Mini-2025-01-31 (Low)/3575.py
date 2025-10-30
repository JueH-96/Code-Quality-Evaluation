from typing import List
from collections import defaultdict

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[j, a, b] = True means we have a subsequence (so far) with j elements selected
        # where if j < k then a is the OR of these j numbers (left half in progress),
        # and if j >= k then a is the OR of the first k numbers (i.e. left half)
        # and b is the OR of the numbers selected after the first k (i.e. right half).
        dp = defaultdict(bool)
        dp[(0, 0, 0)] = True

        for num in nums:
            new_dp = dp.copy()  # when skipping, we keep previously computed states.
            # iterate over all current states and try to select current num if we need more elements.
            for (count, left_or, right_or) in dp:
                # if we have already selected 2*k, we cannot select more.
                if count == 2 * k:
                    continue
                # Option: select the current number.
                if count < k:
                    # still building the left half.
                    new_state = (count + 1, left_or | num, right_or)
                else: 
                    # building right half. count is between k and 2*k-1 inclusive.
                    new_state = (count + 1, left_or, right_or | num)
                new_dp[new_state] = True
            dp = new_dp

        ans = 0
        # Only consider states with exactly 2*k elements selected.
        for (count, a, b) in dp:
            if count == 2 * k:
                ans = max(ans, a ^ b)
        return ans