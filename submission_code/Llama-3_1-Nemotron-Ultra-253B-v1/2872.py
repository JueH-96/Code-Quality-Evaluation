from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        current_sum = nums[-1]
        max_sum = current_sum
        for i in range(len(nums)-2, -1, -1):
            if nums[i] <= current_sum:
                current_sum += nums[i]
            else:
                current_sum = nums[i]
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum