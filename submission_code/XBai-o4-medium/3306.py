import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        total_sum = sum(nums)
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        answer = []
        
        for index, k in queries:
            # Step 1: Mark the specified index if not already marked
            if not marked[index]:
                marked[index] = True
                total_sum -= nums[index]
            
            # Step 2: Mark up to k smallest unmarked elements
            count = 0
            while count < k and heap:
                val, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    total_sum -= val
                    count += 1
            
            answer.append(total_sum)
        
        return answer