from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        freq = sorted(count.values())
        min_size = min(freq)
        
        def can_form_groups(group_size):
            total_groups = 0
            for f in freq:
                full_groups = f // (group_size + 1)
                remainder = f % (group_size + 1)
                if remainder > group_size:
                    return False
                total_groups += full_groups + (1 if remainder else 0)
            return True
        
        for group_size in range(min_size, 0, -1):
            if can_form_groups(group_size):
                return sum((f + group_size) // (group_size + 1) for f in freq)
        
        return len(nums)