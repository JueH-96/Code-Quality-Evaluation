from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        min_val = float('inf')
        for i in range(len(nums) - 1):
            min_val = min(min_val, nums[i+1] - nums[i])
        return min_val