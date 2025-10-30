import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []  # min-heap storing negative distances
        results = []
        for query in queries:
            x, y = query
            dist = abs(x) + abs(y)
            if len(heap) < k:
                heapq.heappush(heap, -dist)
            else:
                if dist < -heap[0]:  # dist < max distance in heap
                    heapq.heappop(heap)
                    heapq.heappush(heap, -dist)
            # After heap adjustment, determine the result
            if len(heap) < k:
                results.append(-1)
            else:
                results.append(-heap[0])  # negative of the negative distance gives the actual distance
        return results