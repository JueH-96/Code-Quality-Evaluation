from typing import List
from sortedcontainers import SortedList

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        queries_with_index = sorted(enumerate(queries), key=lambda x: x[1][0])
        nums = sorted(zip(nums1, nums2), key=lambda x: x[0])
        result = [-1] * len(queries)
        sorted_list = SortedList()
        
        j = 0
        for i, (x, y) in queries_with_index:
            while j < len(nums) and nums[j][0] >= x:
                sorted_list.add(nums[j][1])
                j += 1
            while sorted_list and sorted_list[0] < y:
                sorted_list.remove(sorted_list[0])
            if sorted_list:
                result[i] = max(result[i], nums[j-1][0] + sorted_list[-1])
        
        return result