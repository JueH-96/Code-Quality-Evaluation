from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        if not nums:
            return 0  # Though constraints ensure len(nums) >= 1, this handles edge case.
        current_sum = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                current_sum = nums[i]
        return current_sum