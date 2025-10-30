from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        
        # Build prefix OR array
        for i in range(n):
            prefix[i + 1] = prefix[i] | nums[i]
        
        # Build suffix OR array
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i]
        
        max_or = 0
        for i in range(n):
            # OR of all elements except nums[i]
            left = prefix[i]
            right = suffix[i + 1]
            total = left | right
            # Apply k shifts to nums[i]
            shifted = nums[i] << k
            current_or = total | shifted
            if current_or > max_or:
                max_or = current_or
        
        return max_or