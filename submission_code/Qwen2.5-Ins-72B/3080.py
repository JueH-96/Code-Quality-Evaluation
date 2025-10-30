from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        total_and = nums[0]
        for num in nums:
            total_and &= num
        
        if total_and != 0:
            return 1
        
        count = 0
        current_and = nums[0]
        
        for i in range(len(nums)):
            current_and &= nums[i]
            if current_and == 0:
                count += 1
                if i + 1 < len(nums):
                    current_and = nums[i + 1]
        
        return count