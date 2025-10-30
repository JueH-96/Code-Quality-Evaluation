from typing import List
import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Build a min-heap of (value, index) for all elements.
        heap = [(val, i) for i, val in enumerate(nums)]
        heapq.heapify(heap)
        
        marked = [False] * n
        unmarked_sum = sum(nums)
        result = []
        
        for index, k in queries:
            # Step 1: mark the element at the given index if not already marked.
            if not marked[index]:
                marked[index] = True
                unmarked_sum -= nums[index]
            
            # Step 2: mark k unmarked elements with smallest (value, index).
            while k > 0 and heap:
                # Remove elements from the heap that have already been marked.
                while heap and marked[heap[0][1]]:
                    heapq.heappop(heap)
                if not heap:
                    break
                val, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    unmarked_sum -= val
                    k -= 1
            
            result.append(unmarked_sum)
        
        return result