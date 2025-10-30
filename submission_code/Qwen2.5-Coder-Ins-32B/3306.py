from typing import List
import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        total_sum = sum(nums)
        marked = set()
        min_heap = [(val, idx) for idx, val in enumerate(nums)]
        heapq.heapify(min_heap)
        
        result = []
        
        for index_i, k_i in queries:
            if index_i not in marked:
                marked.add(index_i)
                total_sum -= nums[index_i]
            
            while k_i > 0 and min_heap:
                val, idx = heapq.heappop(min_heap)
                if idx not in marked:
                    marked.add(idx)
                    total_sum -= val
                    k_i -= 1
            
            result.append(total_sum)
        
        return result