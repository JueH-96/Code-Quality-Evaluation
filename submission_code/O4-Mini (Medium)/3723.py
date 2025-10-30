from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        n = len(nums)
        
        for i in range(n):
            left_idx = i - k
            right_idx = i + k
            
            is_good = True
            
            # Check left neighbor if it exists
            if left_idx >= 0 and nums[i] <= nums[left_idx]:
                is_good = False
            
            # Check right neighbor if it exists
            if right_idx < n and nums[i] <= nums[right_idx]:
                is_good = False
            
            if is_good:
                total += nums[i]
        
        return total