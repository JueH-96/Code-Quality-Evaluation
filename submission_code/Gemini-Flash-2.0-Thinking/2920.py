import math
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if len(set(nums)) == 1:
            return 0

        unique_nums = sorted(list(set(nums)))
        min_seconds = float('inf')

        for target in unique_nums:
            target_indices = sorted([i for i, num in enumerate(nums) if num == target])

            if not target_indices:
                continue

            if len(target_indices) == n:
                return 0

            max_time = 0
            num_targets = len(target_indices)

            for i in range(num_targets):
                current_index = target_indices[i]
                next_index = target_indices[(i + 1) % num_targets]

                gap = (next_index - current_index - 1 + n) % n
                time = math.ceil(gap / 2)
                max_time = max(max_time, time)

            min_seconds = min(min_seconds, max_time)

        return min_seconds