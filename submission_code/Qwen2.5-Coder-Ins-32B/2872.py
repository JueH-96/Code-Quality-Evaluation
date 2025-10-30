from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n - 2, -1, -1):
            if nums[i] <= nums[i + 1]:
                nums[i + 1] += nums[i]
        return max(nums)