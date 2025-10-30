from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        total_and = nums[0]
        for num in nums:
            total_and &= num
        
        if total_and != 0:
            return 1
        
        count, current_and = 0, -1
        for num in nums:
            if current_and == -1:
                current_and = num
            else:
                current_and &= num
            if current_and == 0:
                count += 1
                current_and = -1
        
        return count