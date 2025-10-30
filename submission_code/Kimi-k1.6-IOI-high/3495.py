import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        for i, (x, y) in enumerate(queries):
            d = abs(x) + abs(y)
            if len(heap) < k:
                heapq.heappush(heap, -d)
            else:
                current_max = -heap[0]
                if d < current_max:
                    heapq.heappop(heap)
                    heapq.heappush(heap, -d)
            if (i + 1) >= k:
                results.append(-heap[0])
            else:
                results.append(-1)
        return results