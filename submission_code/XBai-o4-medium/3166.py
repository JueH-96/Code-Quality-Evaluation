from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        freq = list(Counter(nums).values())
        min_freq = min(freq)
        low = 1
        high = min_freq
        best_s = 1  # Initialize with the minimal possible s
        
        while low <= high:
            mid = (low + high) // 2
            valid = True
            for f in freq:
                lower_x = (f + mid) // (mid + 1)
                upper_x = f // mid
                if lower_x > upper_x:
                    valid = False
                    break
            if valid:
                best_s = mid
                low = mid + 1
            else:
                high = mid - 1
        
        total = 0
        s = best_s
        for f in freq:
            lower_x = (f + s) // (s + 1)
            total += lower_x
        
        return total