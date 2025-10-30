import heapq
from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        # Create a min heap from the input list. Python's heapq module provides a min heap implementation.
        # The heappush function pushes elements onto the heap, maintaining the heap invariant.
        # The heappop function pops and returns the smallest element from the heap.
        heap = nums
        heapq.heapify(heap)

        # Perform k operations on the heap.
        for _ in range(k):
            # Pop the smallest element from the heap.
            min_element = heapq.heappop(heap)
            # Multiply the element by the multiplier and push it back onto the heap.
            heapq.heappush(heap, min_element * multiplier)

        # Convert the heap back to a list and return it.
        return list(heap)