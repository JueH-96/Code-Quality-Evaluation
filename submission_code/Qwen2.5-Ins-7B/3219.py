from typing import List
from sortedcontainers import SortedList

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        sorted_nums = SortedList(nums)
        groups = []
        start = 0
        
        for i in range(1, n):
            if sorted_nums[i] - sorted_nums[start] > limit:
                groups.append(sorted_nums.irange(sorted_nums[start], sorted_nums[i - 1]))
                start = i
        
        groups.append(sorted_nums.irange(sorted_nums[start], sorted_nums[n - 1]))
        
        result = [0] * n
        for i, num in enumerate(nums):
            for group in groups:
                if num in group:
                    result[i] = group[0]
                    group.pop(0)
                    break
        
        return result