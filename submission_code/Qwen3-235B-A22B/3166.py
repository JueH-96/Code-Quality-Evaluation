from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        counts = list(Counter(nums).values())
        if not counts:
            return 0  # Should not happen as per constraints
        
        # Check if any count is 1, which forces k to be 1
        if any(c == 1 for c in counts):
            k = 1
            total = 0
            for c in counts:
                total += (c + k) // (k + 1)
            return total
        
        min_count = min(counts)
        
        def is_valid(k):
            for c in counts:
                upper = c // k
                lower = (c + k) // (k + 1)
                if lower > upper:
                    return False
            return True
        
        # Binary search to find the best k
        low, high = 1, min_count
        best_k = 1
        
        while low <= high:
            mid = (low + high) // 2
            if is_valid(mid):
                best_k = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Calculate the total groups using best_k
        total = 0
        for c in counts:
            total += (c + best_k) // (best_k + 1)
        return total