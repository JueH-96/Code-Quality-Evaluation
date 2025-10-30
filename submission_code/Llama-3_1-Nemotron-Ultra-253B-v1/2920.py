from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        pos_dict = defaultdict(list)
        for i, num in enumerate(nums):
            pos_dict[num].append(i)
        
        min_steps = float('inf')
        for num in pos_dict:
            positions = pos_dict[num]
            m = len(positions)
            max_gap = 0
            for i in range(m):
                current = positions[i]
                next_p = positions[(i + 1) % m]
                if next_p > current:
                    gap = next_p - current - 1
                else:
                    gap = (next_p + n - current - 1)
                if gap > max_gap:
                    max_gap = gap
            candidate = (max_gap + 1) // 2
            if candidate < min_steps:
                min_steps = candidate
        return min_steps