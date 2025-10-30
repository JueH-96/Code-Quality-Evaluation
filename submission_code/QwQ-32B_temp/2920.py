from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        pos_dict = defaultdict(list)
        for i, num in enumerate(nums):
            pos_dict[num].append(i)
        
        min_time = float('inf')
        
        for x in pos_dict:
            pos = pos_dict[x]
            m = len(pos)
            if m == n:
                current_time = 0
            else:
                if m == 1:
                    max_gap = n - 1
                else:
                    max_gap = 0
                    prev = pos[0]
                    current_max = 0
                    for i in range(1, m):
                        current = pos[i]
                        gap = current - prev - 1
                        if gap > current_max:
                            current_max = gap
                        prev = current
                    # Compute the circular gap between last and first
                    gap_circular = (pos[0] + n - pos[-1] - 1)
                    if gap_circular > current_max:
                        current_max = gap_circular
                    max_gap = current_max
                current_time = (max_gap + 1) // 2
            if current_time < min_time:
                min_time = current_time
        
        return min_time