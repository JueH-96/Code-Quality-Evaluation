from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        prefix_or = [0] * n
        suffix_or = [0] * n
        
        # Compute prefix OR
        prefix_or[0] = nums[0]
        for i in range(1, n):
            prefix_or[i] = prefix_or[i-1] | nums[i]
        
        # Compute suffix OR
        suffix_or[-1] = nums[-1]
        for i in range(n-2, -1, -1):
            suffix_or[i] = suffix_or[i+1] | nums[i]
        
        max_or = 0
        for i in range(n):
            # Calculate OR without nums[i]
            left = prefix_or[i-1] if i > 0 else 0
            right = suffix_or[i+1] if i < n-1 else 0
            current_or_without = left | right
            
            # Calculate current OR when nums[i] is shifted k times
            shifted_num = nums[i] << k
            current_or = shifted_num | current_or_without
            if current_or > max_or:
                max_or = current_or
        
        return max_or