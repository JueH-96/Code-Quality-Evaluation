from collections import Counter
from math import ceil
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        min_freq = min(count.values())
        for k in range(min_freq, 0, -1):
            total_groups = 0
            valid = True
            for freq in count.values():
                groups = ceil(freq / (k + 1))
                if groups * k > freq:
                    valid = False
                    break
                total_groups += groups
            if valid:
                return total_groups
        return len(nums)