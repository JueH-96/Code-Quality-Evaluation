from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        nums.sort()
        for i in range(n):
            if nums[i] < i:
                return False
            if nums[i] > nums[i - 1] + nums[i - 2] + 1:
                return False
        return True