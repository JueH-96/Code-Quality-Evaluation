from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp_add_prev = nums[0]
        dp_sub_prev = float('-inf')
        dp_max = nums[0]
        for num in nums[1:]:
            dp_add = dp_max + num
            dp_sub = dp_add_prev - num
            dp_max_new = max(dp_max, dp_add, dp_sub)
            dp_add_prev, dp_sub_prev = dp_add, dp_sub
            dp_max = dp_max_new
        return dp_max