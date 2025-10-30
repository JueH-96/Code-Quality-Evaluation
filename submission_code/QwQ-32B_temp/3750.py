from bisect import bisect_left
from collections import defaultdict
from typing import List

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        index_map = defaultdict(list)
        n = len(nums)
        for i, num in enumerate(nums):
            index_map[num].append(i)
        
        res = []
        for q in queries:
            val = nums[q]
            indices = index_map.get(val, [])
            if len(indices) <= 1:
                res.append(-1)
                continue
            pos = bisect_left(indices, q)
            # Calculate previous and next indices considering circular array
            prev = indices[pos - 1] if pos > 0 else indices[-1]
            next_ = indices[pos + 1] if pos < len(indices) - 1 else indices[0]
            
            # Compute distances
            d1 = min(abs(q - prev), n - abs(q - prev))
            d2 = min(abs(q - next_), n - abs(q - next_))
            min_d = min(d1, d2)
            res.append(min_d)
        
        return res