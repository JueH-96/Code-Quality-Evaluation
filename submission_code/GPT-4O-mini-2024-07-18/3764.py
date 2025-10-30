from typing import List
import heapq

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        # Create a max heap to store the largest elements
        max_heap = []
        
        # Collect the largest elements from each row based on the limits
        for i in range(len(grid)):
            # Get the largest elements from the current row
            largest_elements = heapq.nlargest(limits[i], grid[i])
            # Add them to the max heap
            for element in largest_elements:
                heapq.heappush(max_heap, element)
        
        # Now we need to take the largest k elements from the max heap
        max_sum = 0
        for _ in range(min(k, len(max_heap))):
            max_sum += heapq.heappop(max_heap)
        
        return max_sum