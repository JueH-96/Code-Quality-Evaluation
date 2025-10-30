import math
from typing import List
from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        pos_dict = defaultdict(list)
        for i, num in enumerate(nums):
            pos_dict[num].append(i)
        
        min_time = float('inf')
        
        for pos_list in pos_dict.values():
            pos = sorted(pos_list)
            k = len(pos)
            max_min_dist = 0
            
            # Compute gaps between consecutive positions
            for i in range(k - 1):
                gap = pos[i + 1] - pos[i]
                min_dist_in_gap = gap // 2
                if min_dist_in_gap > max_min_dist:
                    max_min_dist = min_dist_in_gap
            
            # Compute wrap-around gap
            wrap_gap = (pos[0] + n - pos[-1])
            min_dist_wrap = wrap_gap // 2
            if min_dist_wrap > max_min_dist:
                max_min_dist = min_dist_wrap
            
            # Update minimum time
            if max_min_dist < min_time:
                min_time = max_min_dist
        
        return min_time