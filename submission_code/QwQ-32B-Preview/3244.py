import math
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        if not nums:
            return 0  # Edge case: empty array
        
        min_val = min(nums)
        f = nums.count(min_val)
        
        # Calculate ceil(f / 2) + (f % 2)
        min_length = math.ceil(f / 2) + (f % 2)
        
        # Ensure the minimum length is at least 1
        return max(1, min_length)