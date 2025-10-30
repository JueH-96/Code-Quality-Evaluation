from typing import List
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        def is_possible(g, freqs, n):
            group_size = (n + g - 1) // g  # ceil(n / g)
            total_groups_needed = 0
            for f in freqs:
                total_groups_needed += (f + group_size - 1) // group_size  # ceil(f / group_size)
                if total_groups_needed > g:
                    return False
            return total_groups_needed <= g
        
        freqs = list(Counter(nums).values())
        n = len(nums)
        low, high = 1, n
        while low < high:
            mid = (low + high) // 2
            if is_possible(mid, freqs, n):
                high = mid
            else:
                low = mid + 1
        return low