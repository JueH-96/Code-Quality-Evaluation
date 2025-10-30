from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        while len(nums) >= 3:
            if nums[-1] < sum(nums[:-1]):
                return sum(nums)
            nums.pop()
        return -1