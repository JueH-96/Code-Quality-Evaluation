from typing import List
from collections import deque

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        indexed_nums = sorted((num, idx) for idx, num in enumerate(nums))
        result = [0] * n
        i = 0
        
        while i < n:
            start = i
            current = indexed_nums[i][0]
            indices = []
            
            while i < n and indexed_nums[i][0] <= current + limit:
                indices.append(indexed_nums[i][1])
                i += 1
            
            sorted_indices = sorted(indices)
            for j, idx in enumerate(sorted_indices):
                result[idx] = indexed_nums[start + j][0]
        
        return result