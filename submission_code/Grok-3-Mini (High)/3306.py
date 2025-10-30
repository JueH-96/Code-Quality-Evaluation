import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Create a min-heap with (value, index)
        heap = [(num, i) for i, num in enumerate(nums)]
        heapq.heapify(heap)
        # Marked array to keep track of marked indices
        marked = [False] * n
        # Total sum of unmarked elements
        total_sum = sum(nums)
        # Answer list to store sum after each query
        ans = []
        
        for index_i, k_i in queries:
            # Mark the element at index_i if not already marked
            if not marked[index_i]:
                marked[index_i] = True
                total_sum -= nums[index_i]
            
            # Mark up to k_i unmarked elements with smallest values and indices
            k = k_i
            while k > 0 and heap:
                val, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    total_sum -= val
                    k -= 1
                # If marked, discard and continue
            
            # Append the current sum of unmarked elements
            ans.append(total_sum)
        
        return ans