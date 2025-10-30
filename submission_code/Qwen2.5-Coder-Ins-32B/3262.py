from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        total_sum = sum(nums)
        
        for num in reversed(nums):
            if total_sum - num > num:
                return total_sum
            total_sum -= num
        
        return -1