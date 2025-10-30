from heapq import heappush, heappop
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        for i, (x, y) in enumerate(queries):
            dist = abs(x) + abs(y)
            current_obstacles = i + 1
            if current_obstacles < k:
                heappush(heap, -dist)
                results.append(-1)
            else:
                if len(heap) < k:
                    heappush(heap, -dist)
                else:
                    if dist < (-heap[0]):
                        heappop(heap)
                        heappush(heap, -dist)
                results.append(-heap[0])
        return results