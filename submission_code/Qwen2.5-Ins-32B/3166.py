from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        freq = count.values()
        min_freq = min(freq)
        
        for size in range(min_freq + 1, 0, -1):
            total_groups = 0
            valid = True
            for f in freq:
                groups, remainder = divmod(f, size)
                if remainder > groups:
                    valid = False
                    break
                total_groups += groups + (1 if remainder else 0)
            if valid:
                return total_groups