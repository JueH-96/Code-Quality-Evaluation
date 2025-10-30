from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        total = sum(nums)
        while len(nums) >= 3:
            if nums[-1] < total - nums[-1]:
                return total
            else:
                removed = nums.pop()
                total -= removed
        return -1