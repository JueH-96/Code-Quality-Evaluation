import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        total = sum(nums)
        heap = [ (nums[i], i) for i in range(n) ]
        heapq.heapify(heap)
        
        marked = [False] * n
        answer = []
        
        for index_i, k_i in queries:
            # Mark the index_i if not already marked
            if not marked[index_i]:
                marked[index_i] = True
                total -= nums[index_i]
            
            # Now process the k_i elements
            count = 0
            while count < k_i and heap:
                val, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    total -= val
                    count += 1
            
            answer.append(total)
        
        return answer