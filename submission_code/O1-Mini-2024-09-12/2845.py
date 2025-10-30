from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        min_diff = float('inf')
        for i in range(len(nums) - 1):
            diff = abs(nums[i] - nums[i + 1])
            if diff < min_diff:
                min_diff = diff
        return min_diff