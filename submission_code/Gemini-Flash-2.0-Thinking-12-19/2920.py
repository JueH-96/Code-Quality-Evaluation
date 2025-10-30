import math

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if len(set(nums)) == 1:
            return 0
        unique_values = sorted(list(set(nums)))
        min_seconds = float('inf')
        for target_value in unique_values:
            target_indices = [i for i, x in enumerate(nums) if x == target_value]
            if not target_indices:
                continue
            if len(target_indices) == n:
                min_seconds = min(min_seconds, 0)
                continue
            max_gap = 0
            if len(target_indices) == 1:
                max_gap = n - 1
            else:
                sorted_indices = sorted(target_indices)
                gaps = []
                for i in range(len(sorted_indices) - 1):
                    gaps.append(sorted_indices[i+1] - sorted_indices[i] - 1)
                gaps.append((sorted_indices[0] + n - sorted_indices[-1] - 1) % n)
                max_gap = max(gaps) if gaps else 0
            seconds_needed = math.ceil(max_gap / 2)
            min_seconds = min(min_seconds, seconds_needed)
        return min_seconds

from typing import List