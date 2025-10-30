from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        current_or = 0
        left = 0
        
        for right in range(len(nums)):
            current_or |= nums[right]
            while left <= right and current_or >= k:
                min_diff = min(min_diff, abs(current_or - k))
                current_or &= ~nums[left]
                left += 1
            
            min_diff = min(min_diff, abs(current_or - k))
        
        return min_diff