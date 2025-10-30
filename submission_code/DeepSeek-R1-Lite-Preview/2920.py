from typing import List
from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        positions = defaultdict(list)
        for idx, num in enumerate(nums):
            positions[num].append(idx)
        
        min_time = float('inf')
        
        for value, pos_list in positions.items():
            k = len(pos_list)
            if k == n:
                min_time = 0
                break
            if k == 1:
                gap = n
            else:
                max_gap = 0
                for i in range(k):
                    current = pos_list[i]
                    next_pos = pos_list[(i + 1) % k]
                    if next_pos > current:
                        gap = next_pos - current - 1
                    else:
                        gap = (n - current) + next_pos - 1
                    max_gap = max(max_gap, gap)
                gap = max_gap
            time = (gap + 1) // 2
            min_time = min(min_time, time)
        
        return min_time