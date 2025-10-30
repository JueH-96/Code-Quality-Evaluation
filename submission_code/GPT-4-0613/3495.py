import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        for query in queries:
            distance = abs(query[0]) + abs(query[1])
            heapq.heappush(heap, -distance)
            if len(heap) < k:
                results.append(-1)
            else:
                while len(heap) > k:
                    heapq.heappop(heap)
                results.append(-heap[0])
        return results