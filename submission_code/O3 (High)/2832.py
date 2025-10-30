from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Gather the positions (indices) for every distinct value
        positions = defaultdict(list)
        for idx, value in enumerate(nums):
            positions[value].append(idx)
        
        best = 0
        
        # For every value, use the sliding–window technique described
        for idx_list in positions.values():
            left = 0
            for right, r_idx in enumerate(idx_list):
                # required deletions inside the window [left, right]
                # equals #non-value elements in that interval:
                #   (r_idx - idx_list[left]) - (right - left)
                while r_idx - idx_list[left] - (right - left) > k:
                    left += 1
                best = max(best, right - left + 1)
        
        return best