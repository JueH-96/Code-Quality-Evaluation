class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        import math
        
        # Count frequencies of each unique value
        counts = list(Counter(nums).values())
        
        min_groups = float('inf')
        
        # Try each possible base size from 1 to min(counts)
        for s in range(1, min(counts) + 1):
            total_groups = 0
            valid = True
            for c in counts:
                # Check if we can partition c into groups of size s or s+1
                min_g = math.ceil(c / (s + 1))
                max_g = c // s
                if min_g > max_g:
                    valid = False
                    break
                total_groups += min_g
            
            if valid:
                min_groups = min(min_groups, total_groups)
        
        return min_groups