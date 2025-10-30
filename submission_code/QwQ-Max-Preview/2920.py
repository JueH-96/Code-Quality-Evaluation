from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        val_to_indices = defaultdict(list)
        for i, num in enumerate(nums):
            val_to_indices[num].append(i)
        
        min_time = float('inf')
        for val, indices in val_to_indices.items():
            m = len(indices)
            if m == 0:
                continue  # Not possible given the problem constraints
            max_gap = 0
            # Compute gaps between consecutive indices
            for i in range(1, m):
                gap = indices[i] - indices[i-1] - 1
                current = (gap + 1) // 2  # Equivalent to ceil(gap / 2)
                if current > max_gap:
                    max_gap = current
            # Compute wrap-around gap
            wrap_gap = (indices[0] + n - indices[-1] - 1)
            current = (wrap_gap + 1) // 2
            if current > max_gap:
                max_gap = current
            # Update the minimum time
            if max_gap < min_time:
                min_time = max_gap
        return min_time