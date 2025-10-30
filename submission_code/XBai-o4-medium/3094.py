from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        total_ops = 0
        
        for v in counts.values():
            if v == 1:
                return -1
            elif v % 3 == 0:
                total_ops += v // 3
            elif v % 3 == 1:
                # Use two operations of 2 (e.g., 4 becomes 2+2)
                total_ops += (v + 2) // 3
            else:  # v % 3 == 2
                # Use one operation of 2 (e.g., 5 becomes 3+2)
                total_ops += (v + 1) // 3
        
        return total_ops