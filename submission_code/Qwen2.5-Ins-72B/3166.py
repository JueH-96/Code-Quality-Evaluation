from collections import Counter
from math import ceil
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        freq = list(count.values())
        min_freq = min(freq)
        
        for group_size in range(min_freq, 0, -1):
            total_groups = 0
            for f in freq:
                if f % (group_size + 1) == 0:
                    total_groups += f // (group_size + 1)
                elif f % (group_size + 1) <= group_size:
                    total_groups += ceil(f / (group_size + 1))
                else:
                    break
            else:
                return total_groups
        
        return len(nums)