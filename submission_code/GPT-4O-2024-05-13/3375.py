import heapq
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Min-heap to store the multiples of each coin
        min_heap = []
        
        # Initialize the heap with the first multiple of each coin
        for coin in coins:
            heapq.heappush(min_heap, coin)
        
        # Variable to store the current smallest amount
        current_smallest = 0
        
        # Extract the smallest element from the heap k times
        for _ in range(k):
            current_smallest = heapq.heappop(min_heap)
            # Push the next multiple of the extracted coin back into the heap
            for coin in coins:
                if current_smallest % coin == 0:
                    heapq.heappush(min_heap, current_smallest + coin)
                    break
        
        return current_smallest