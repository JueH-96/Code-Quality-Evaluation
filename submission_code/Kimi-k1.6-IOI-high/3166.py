from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        freqs = list(Counter(nums).values())
        if not freqs:
            return 0
        
        max_freq = max(freqs)
        low, high = 1, max_freq
        best_m = 1
        
        # Binary search to find the maximum valid m
        while low <= high:
            mid = (low + high) // 2
            valid = True
            for f in freqs:
                g_min = (f + mid) // (mid + 1)
                g_max = f // mid
                if g_min > g_max:
                    valid = False
                    break
            if valid:
                best_m = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Calculate the total number of groups using best_m
        total_groups = 0
        for f in freqs:
            total_groups += (f + best_m) // (best_m + 1)
        
        return total_groups