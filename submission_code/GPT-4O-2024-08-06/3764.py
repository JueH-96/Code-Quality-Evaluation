from typing import List
import heapq

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        # Create a max heap to store the potential elements to be selected
        max_heap = []
        
        # Iterate over each row in the grid
        for i in range(len(grid)):
            # Sort the row in descending order to get the largest elements first
            sorted_row = sorted(grid[i], reverse=True)
            # Take the top 'limits[i]' elements from this sorted row
            for j in range(min(limits[i], len(sorted_row))):
                # Push the negative of the element to simulate a max heap using heapq
                heapq.heappush(max_heap, -sorted_row[j])
        
        # We need to select at most 'k' elements from the heap
        max_sum = 0
        for _ in range(min(k, len(max_heap))):
            # Pop the largest element from the heap (remember to negate it back)
            max_sum += -heapq.heappop(max_heap)
        
        return max_sum