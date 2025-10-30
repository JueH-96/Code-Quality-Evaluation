from typing import List
from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)

        # Collect indices for every distinct value
        positions = defaultdict(list)
        for idx, val in enumerate(nums):
            positions[val].append(idx)

        best = n  # upper bound, can't be larger than n

        for idx_list in positions.values():
            max_gap = 0

            # gaps between consecutive occurrences
            for i in range(1, len(idx_list)):
                gap = idx_list[i] - idx_list[i - 1] - 1
                if gap > max_gap:
                    max_gap = gap

            # circular gap between last and first occurrence
            circular_gap = idx_list[0] + n - idx_list[-1] - 1
            if circular_gap > max_gap:
                max_gap = circular_gap

            # seconds needed for this value
            seconds = (max_gap + 1) // 2
            if seconds < best:
                best = seconds

        return best