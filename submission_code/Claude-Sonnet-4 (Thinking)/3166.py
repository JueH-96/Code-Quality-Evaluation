class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        import math
        
        freq = Counter(nums)
        frequencies = list(freq.values())
        
        # Try k from largest possible down to 1 (larger k gives fewer groups)
        for k in range(min(frequencies), 0, -1):
            valid = True
            total_groups = 0
            
            for f in frequencies:
                min_groups_for_f = math.ceil(f / (k + 1))
                max_groups_for_f = f // k
                
                # Check if we can split frequency f into groups of size k or k+1
                if min_groups_for_f > max_groups_for_f:
                    valid = False
                    break
                
                total_groups += min_groups_for_f
            
            if valid:
                return total_groups