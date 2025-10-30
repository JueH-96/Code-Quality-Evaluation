from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Initialize answer to a large number.
        ans = float('inf')
        # The first subarray always starts at index 0.
        # We need to choose two boundaries: idx1 (end of first subarray) and idx2 (end of second subarray)
        # such that:
        #   0 <= idx1 < idx2 < n-1.
        # The cost is then: nums[0] + nums[idx1 + 1] + nums[idx2 + 1].
        for i in range(0, n - 2):
            for j in range(i + 1, n - 1):
                cost = nums[0] + nums[i + 1] + nums[j + 1]
                ans = min(ans, cost)
        return ans