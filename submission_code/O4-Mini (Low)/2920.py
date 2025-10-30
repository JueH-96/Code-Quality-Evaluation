from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        # Collect positions for each value
        pos_map = defaultdict(list)
        for i, v in enumerate(nums):
            pos_map[v].append(i)
        
        ans = float('inf')
        # For each value, compute the maximal gap between consecutive occurrences (circularly)
        for positions in pos_map.values():
            positions.sort()
            # Track the largest circular gap D between sources
            max_gap = 0
            m = len(positions)
            # Consecutive gaps
            for i in range(m - 1):
                gap = positions[i+1] - positions[i]
                if gap > max_gap:
                    max_gap = gap
            # Wrap‐around gap
            wrap_gap = positions[0] + n - positions[-1]
            if wrap_gap > max_gap:
                max_gap = wrap_gap
            # Time needed is floor(D/2)
            time_needed = max_gap // 2
            ans = min(ans, time_needed)
        
        return ans