from collections import Counter
from math import ceil
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        num_counts = Counter(nums).values()
        min_group_size = min(num_counts)
        max_group_size = max(num_counts)
        
        def is_valid_groups(group_size):
            # Calculate minimum and maximum groups needed with given group size
            min_groups = ceil(n / group_size) if n % group_size != 0 else n // group_size
            max_groups = n // group_size + 1 if n % group_size != 0 else n // group_size
            total_groups = 0
            for count in num_counts:
                if count < group_size or ceil(count / group_size) > max_groups or count // group_size < min_groups:
                    return False
                total_groups += ceil(count / group_size)
                if total_groups > max_groups * len(num_counts):
                    return False
            return True
        
        # Optimize by checking only up to sqrt(sum(num_counts))
        upper_bound = int(ceil(sum(num_counts)**0.5))
        
        # Find the minimum group size that is valid
        for group_size in range(min_group_size, min(max_group_size, upper_bound) + 1):
            if is_valid_groups(group_size):
                return ceil(sum(ceil(count / group_size) for count in num_counts))
        
        # Fallback to distributing evenly or as close as possible
        return ceil(sum(num_counts) / len(num_counts))