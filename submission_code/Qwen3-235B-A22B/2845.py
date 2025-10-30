from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        min_diff = float('inf')
        for i in range(len(nums) - 1):
            current_diff = nums[i+1] - nums[i]
            if current_diff < min_diff:
                min_diff = current_diff
        return min_diff