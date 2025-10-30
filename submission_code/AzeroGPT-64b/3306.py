import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n  # Track marked indices
        unmarked_heap = [(val, idx) for idx, val in enumerate(nums)]  # Heap with unmarked values and their indices
        heapq.heapify(unmarked_heap)
        result = []
        total = sum(nums)

        for index, k in queries:
            if not marked[index]:  # Mark if not already marked
                marked[index] = True
                total -= nums[index]
            
            while k > 0 and unmarked_heap:  # Mark k smallest unmarked elements
                _, i = heapq.heappop(unmarked_heap)
                if not marked[i]:  # Mark if not already marked
                    marked[i] = True
                    total -= nums[i]
                    k -= 1
            
            result.append(total)
        
        return result