from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Compute the overall AND of the entire array
        x = nums[0]
        for num in nums[1:]:
            x &= num
        
        # If the overall AND is non-zero, the only possible split is the entire array
        if x != 0:
            return 1
        
        count = 0
        current_and = None
        
        for num in nums:
            if current_and is None:
                current_and = num
            else:
                current_and &= num
            
            # When current_and becomes zero, split here
            if current_and == 0:
                count += 1
                current_and = None  # Reset for the next subarray
        
        # If all elements are covered by splits, return count; otherwise, return 1
        return count if current_and is None else 1