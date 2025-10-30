import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        count = 0
        for x, y in queries:
            d = abs(x) + abs(y)
            count += 1
            if len(heap) < k:
                heapq.heappush(heap, -d)
            else:
                current_max = -heap[0]
                if d < current_max:
                    heapq.heappop(heap)
                    heapq.heappush(heap, -d)
            if count >= k:
                results.append(-heap[0])
            else:
                results.append(-1)
        return results