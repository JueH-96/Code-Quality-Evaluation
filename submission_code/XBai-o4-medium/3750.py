from collections import defaultdict
from typing import List

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        pos = defaultdict(list)
        for i, num in enumerate(nums):
            pos[num].append(i)
        
        min_dist = [-1] * n
        
        for key in pos:
            indices = pos[key]
            m = len(indices)
            if m < 2:
                continue
            for k in range(m):
                prev_k = (k - 1 + m) % m
                next_k = (k + 1) % m
                a = indices[k]
                b_prev = indices[prev_k]
                d_prev = abs(a - b_prev)
                d_prev_min = min(d_prev, n - d_prev)
                
                b_next = indices[next_k]
                d_next = abs(a - b_next)
                d_next_min = min(d_next, n - d_next)
                
                min_dist[a] = min(d_prev_min, d_next_min)
        
        return [min_dist[q] for q in queries]