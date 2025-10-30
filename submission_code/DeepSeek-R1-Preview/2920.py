from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        pos_dict = defaultdict(list)
        n = len(nums)
        for i, num in enumerate(nums):
            pos_dict[num].append(i)
        
        if len(pos_dict) == 1:
            return 0  # All elements are already the same
        
        min_time = float('inf')
        for x in pos_dict:
            positions = sorted(pos_dict[x])
            extended = positions + [positions[0] + n]
            max_gap = 0
            for i in range(len(extended) - 1):
                diff = extended[i+1] - extended[i]
                gap = diff - 1
                if gap > max_gap:
                    max_gap = gap
            time = (max_gap + 1) // 2
            if time < min_time:
                min_time = time
        return min_time