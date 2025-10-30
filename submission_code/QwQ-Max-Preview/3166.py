from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums):
        counts = list(Counter(nums).values())
        if not counts:
            return 0
        max_count = max(counts)
        low, high = 1, max_count
        best_a = 1
        
        while low <= high:
            mid = (low + high) // 2
            valid = True
            for c in counts:
                k = (c + mid - 1) // mid  # ceil division
                if (mid - 1) * k > c:
                    valid = False
                    break
            if valid:
                best_a = mid
                low = mid + 1
            else:
                high = mid - 1
        
        total = 0
        for c in counts:
            total += (c + best_a - 1) // best_a
        
        return total