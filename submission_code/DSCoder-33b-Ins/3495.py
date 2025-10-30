from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        for x, y in queries:
            dist = abs(x) + abs(y)
            heapq.heappush(heap, -dist)
            if len(heap) < k:
                results.append(-1)
            elif len(heap) == k:
                results.append(-heap[0])
            else:
                heapq.heappop(heap)
                results.append(-heap[0])
        return results