from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        overall_and = nums[0]
        for num in nums:
            overall_and &= num
        
        if overall_and != 0:
            return 1
        
        count = 0
        current_and = ~0  # Start with all bits set to 1
        
        for num in nums:
            current_and &= num
            if current_and == 0:
                count += 1
                current_and = ~0  # Reset current_and for the next subarray
        
        return count