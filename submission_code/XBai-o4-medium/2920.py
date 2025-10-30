from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        pos = defaultdict(list)
        for i, num in enumerate(nums):
            pos[num].append(i)
        
        min_time = float('inf')
        for x in pos:
            indices = pos[x]
            m = len(indices)
            max_gap = 0
            for i in range(m):
                j = (i + 1) % m
                if j == 0:
                    distance = n - indices[i] + indices[j]
                else:
                    distance = indices[j] - indices[i]
                current_t = distance // 2
                if current_t > max_gap:
                    max_gap = current_t
            if max_gap < min_time:
                min_time = max_gap
        return min_time