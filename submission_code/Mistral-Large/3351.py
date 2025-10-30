from typing import List
import heapq

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Convert the happiness list to a max-heap by negating the values
        max_heap = [-h for h in happiness]
        heapq.heapify(max_heap)

        total_sum = 0

        for _ in range(k):
            # Pop the maximum value from the heap
            max_value = -heapq.heappop(max_heap)
            total_sum += max_value

            # Decrease the happiness value of all remaining children by 1
            for i in range(len(max_heap)):
                max_heap[i] += 1
                if max_heap[i] > 0:
                    max_heap[i] = 0

            heapq.heapify(max_heap)

        return total_sum