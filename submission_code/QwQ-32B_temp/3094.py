from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        total_operations = 0
        
        for count in counts.values():
            if count == 1:
                return -1
            remainder = count % 3
            if remainder == 0:
                total_operations += count // 3
            elif remainder == 1:
                # (count - 4) must be divisible by 3, then add 2 operations (for the two 2's)
                total_operations += (count - 4) // 3 + 2
            else:  # remainder == 2
                total_operations += (count // 3) + 1
        
        return total_operations