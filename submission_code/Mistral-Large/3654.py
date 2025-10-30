from typing import List
import heapq
import math

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Convert nums to a max heap (negating values since heapq is a min-heap by default)
        max_heap = [-num for num in nums]
        heapq.heapify(max_heap)

        # Perform Operation 1
        for _ in range(op1):
            if max_heap:
                largest = -heapq.heappop(max_heap)
                new_value = math.ceil(largest / 2)
                heapq.heappush(max_heap, -new_value)

        # Perform Operation 2
        for _ in range(op2):
            if max_heap:
                largest = -heapq.heappop(max_heap)
                if largest >= k:
                    new_value = largest - k
                else:
                    new_value = largest
                heapq.heappush(max_heap, -new_value)

        # Calculate the sum of the modified array
        return -sum(max_heap)