from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total = 0
        
        for i, val in enumerate(nums):
            is_good = True  # Assume current element is good
            
            # Check left neighbor at distance k
            left_idx = i - k
            if left_idx >= 0 and nums[left_idx] >= val:
                is_good = False    # Not strictly greater than left
           
            # Check right neighbor at distance k
            right_idx = i + k
            if right_idx < n and nums[right_idx] >= val:
                is_good = False    # Not strictly greater than right
            
            if is_good:
                total += val
                
        return total