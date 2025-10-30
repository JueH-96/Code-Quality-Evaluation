from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        freq = Counter(nums)
        frequencies = list(freq.values())
        if not frequencies:
            return 0
        
        max_freq = max(frequencies)
        low, high = 1, max_freq
        best_s = 0
        
        while low <= high:
            mid = (low + high) // 2
            valid = True
            for f in frequencies:
                g = (f + mid) // (mid + 1)  # Equivalent to ceil(f/(mid+1))
                if mid * g > f:
                    valid = False
                    break
            if valid:
                best_s = mid
                low = mid + 1
            else:
                high = mid - 1
        
        s = best_s
        total_groups = 0
        for f in frequencies:
            total_groups += (f + s) // (s + 1)
        
        return total_groups