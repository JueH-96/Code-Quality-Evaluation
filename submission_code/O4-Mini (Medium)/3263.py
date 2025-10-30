from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # nums[0] is always included in the cost
        base = nums[0]
        # We need to pick two split points i, j with 1 <= i < j < n,
        # so that subarrays are [0:i], [i:j], [j:n], all non-empty.
        # Total cost = nums[0] + nums[i] + nums[j]. Minimize nums[i] + nums[j].
        min_pair_sum = float('inf')
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                min_pair_sum = min(min_pair_sum, nums[i] + nums[j])
        return base + min_pair_sum