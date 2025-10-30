import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        total = sum(nums)
        marked = [False] * n
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        
        res = []
        
        for index, k in queries:
            if not marked[index]:
                marked[index] = True
                total -= nums[index]
            
            while k > 0 and heap:
                val, idx = heapq.heappop(heap)
                if marked[idx]:
                    continue
                marked[idx] = True
                total -= val
                k -= 1
            
            res.append(total)
        
        return res