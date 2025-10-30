from typing import List
from heapq import heappush, heappop

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        marked = [False] * len(nums)
        total_sum = sum(nums)
        min_heap = []
        for i, num in enumerate(nums):
            heappush(min_heap, (num, i))
        
        result = []
        for index, k in queries:
            if not marked[index]:
                marked[index] = True
                total_sum -= nums[index]
            
            while k > 0 and min_heap:
                num, i = heappop(min_heap)
                if not marked[i]:
                    marked[i] = True
                    total_sum -= num
                    k -= 1
            
            result.append(total_sum)
        
        return result