from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        count = 1 if nums[0] > 0 else 0
        for i in range(len(nums) - 1):
            if nums[i] < i + 1 < nums[i + 1]:
                count += 1
        if nums[-1] < len(nums):
            count += 1
        return count