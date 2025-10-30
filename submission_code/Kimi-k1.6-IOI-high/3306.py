import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        sum_unmarked = sum(nums)
        marked = [False] * n
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        answer = []
        
        for index_i, k_i in queries:
            # Mark the specified index if not already marked
            if not marked[index_i]:
                marked[index_i] = True
                sum_unmarked -= nums[index_i]
            
            # Mark up to k_i smallest unmarked elements
            count = 0
            while count < k_i and heap:
                val, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    sum_unmarked -= val
                    count += 1
            
            answer.append(sum_unmarked)
        
        return answer