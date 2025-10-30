from collections import deque
from typing import List

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = 0
        q = deque()
        sum_required = 0
        sum_original = 0
        left = 0
        
        for right in range(n):
            current_val = nums[right]
            # Compute the required_val for current_val
            if q:
                prev_required = q[-1]
                required_val = max(prev_required, current_val)
            else:
                required_val = current_val
            
            q.append(required_val)
            sum_required += required_val
            sum_original += current_val
            
            current_increments = sum_required - sum_original
            
            # Adjust the window if increments exceed k
            while current_increments > k:
                removed_val = q.popleft()
                sum_required -= removed_val
                sum_original -= nums[left]
                left += 1
                current_increments = sum_required - sum_original  # Recompute after adjustment
            
            res += right - left + 1
        
        return res