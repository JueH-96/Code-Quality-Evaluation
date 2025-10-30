from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        counts = list(Counter(nums).values())
        if not counts:
            return 0
        
        max_count = max(counts)
        low, high = 1, max_count
        best_x = 1  # default to minimal possible x if no higher x is valid
        
        # Binary search to find the maximum valid x
        while low <= high:
            mid = (low + high) // 2
            valid = True
            for c in counts:
                a = (c + mid) // (mid + 1)  # ceil(c / (mid + 1))
                b = c // mid                # floor(c / mid)
                if a > b:
                    valid = False
                    break
            if valid:
                best_x = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Calculate the minimal number of groups with the best_x
        total = 0
        for c in counts:
            total += (c + best_x) // (best_x + 1)
        
        return total