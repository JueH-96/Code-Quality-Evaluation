from typing import List
from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        num_positions = defaultdict(list)
        
        # Collect positions for each unique value
        for idx, num in enumerate(nums):
            num_positions[num].append(idx)
        
        min_time = float('inf')
        
        # Iterate over each unique value
        for positions in num_positions.values():
            if not positions:
                continue
            # Sort positions to calculate gaps
            positions.sort()
            max_gap = 0
            k = len(positions)
            # Calculate gaps between consecutive positions
            for i in range(k):
                if i == k - 1:
                    # Wrap-around gap
                    gap = (positions[0] + n - positions[i]) % n
                else:
                    gap = positions[i + 1] - positions[i]
                max_gap = max(max_gap, gap)
            # Time to equalize to this value
            time = (max_gap + 1) // 2
            # Update minimum time
            min_time = min(min_time, time)
        
        return min_time