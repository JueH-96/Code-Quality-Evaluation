from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Calculate the total AND of the entire array
        total_and = nums[0]
        for num in nums[1:]:
            total_and &= num
        
        # If the total AND is not zero, we can't split into subarrays with sum zero
        if total_and != 0:
            return 1
        
        current_and = None
        count = 0
        
        for num in nums:
            if current_and is None:
                current_and = num
            else:
                current_and &= num
            
            if current_and == 0:
                count += 1
                current_and = None
        
        return count