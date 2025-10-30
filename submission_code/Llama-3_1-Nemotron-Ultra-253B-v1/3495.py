from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        res = []
        for i, (x, y) in enumerate(queries):
            d = abs(x) + abs(y)
            if len(heap) < k:
                heapq.heappush(heap, -d)
            else:
                if d < -heap[0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, -d)
            res.append(-heap[0] if (i + 1) >= k else -1)
        return res