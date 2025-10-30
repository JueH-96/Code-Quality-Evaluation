from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        d = defaultdict(list)
        for i, val in enumerate(nums):
            d[val].append(i)
        
        min_time = float('inf')
        
        for val in d:
            positions = d[val]
            m = len(positions)
            if m == 0:
                continue  # Not possible as n >= 1
            
            max_gap = 0
            if m == 1:
                max_gap = n
            else:
                max_gap = 0
                for i in range(1, m):
                    gap = positions[i] - positions[i-1]
                    if gap > max_gap:
                        max_gap = gap
                # Check the circular gap
                circular_gap = (n + positions[0] - positions[-1])
                if circular_gap > max_gap:
                    max_gap = circular_gap
            
            current_time = max_gap // 2
            if current_time < min_time:
                min_time = current_time
        
        return min_time