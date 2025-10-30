from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        freqs = list(Counter(nums).values())
        if not freqs:
            return 0
        max_freq = max(freqs)
        low, high = 1, max_freq
        best_s = 1
        
        while low <= high:
            mid = (low + high) // 2
            valid = True
            for f in freqs:
                k = (f + mid - 1) // mid
                if k * (mid - 1) > f:
                    valid = False
                    break
            if valid:
                best_s = mid
                low = mid + 1
            else:
                high = mid - 1
        
        total = 0
        for f in freqs:
            total += (f + best_s - 1) // best_s
        return total