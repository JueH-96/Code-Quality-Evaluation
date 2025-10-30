from typing import List
import collections

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        sorted_nums = sorted((val, idx) for idx, val in enumerate(nums))
        result = [0] * n
        i = 0
        
        while i < n:
            group = []
            group.append(sorted_nums[i])
            while i + 1 < n and sorted_nums[i + 1][0] - sorted_nums[i][0] <= limit:
                group.append(sorted_nums[i + 1])
                i += 1
            i += 1
            
            indices = [idx for val, idx in group]
            values = [val for val, idx in group]
            values.sort()
            
            for j, idx in enumerate(sorted(indices)):
                result[idx] = values[j]
        
        return result