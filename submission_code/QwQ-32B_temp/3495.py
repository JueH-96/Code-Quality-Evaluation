import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        count = 0
        result = []
        for x, y in queries:
            d = abs(x) + abs(y)
            count += 1
            if count <= k:
                heapq.heappush(heap, -d)
            else:
                if d < -heap[0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, -d)
            if count < k:
                result.append(-1)
            else:
                result.append(-heap[0])
        return result