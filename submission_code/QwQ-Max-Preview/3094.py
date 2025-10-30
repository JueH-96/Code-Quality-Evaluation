from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        total_operations = 0
        
        for freq in count.values():
            if freq == 1:
                return -1
            remainder = freq % 3
            if remainder == 0:
                total_operations += freq // 3
            elif remainder == 1:
                total_operations += (freq // 3 - 1) + 2
            else:  # remainder == 2
                total_operations += (freq // 3) + 1
        
        return total_operations