from bisect import bisect_left
from typing import List
from collections import defaultdict

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        index_map = defaultdict(list)
        n = len(nums)
        for i, val in enumerate(nums):
            index_map[val].append(i)
        
        res = []
        for q in queries:
            val = nums[q]
            indices = index_map[val]
            if len(indices) == 1:
                res.append(-1)
                continue
            
            pos = bisect_left(indices, q)
            # Determine left and right neighbors in the circular list
            if pos > 0:
                left = pos - 1
            else:
                left = len(indices) - 1  # wrap around to the last element
            
            if pos < len(indices) - 1:
                right = pos + 1
            else:
                right = 0  # wrap around to the first element
            
            left_val = indices[left]
            right_val = indices[right]
            
            # Calculate distances considering circular array
            d_left = abs(q - left_val)
            d_left_min = min(d_left, n - d_left)
            
            d_right = abs(q - right_val)
            d_right_min = min(d_right, n - d_right)
            
            res.append(min(d_left_min, d_right_min))
        
        return res