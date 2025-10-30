from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        max_heap = []
        results = []
        for x, y in queries:
            dist = abs(x) + abs(y)
            if len(max_heap) < k:
                heapq.heappush(max_heap, -dist)
            else:
                if dist < -max_heap[0]:
                    heapq.heappushpop(max_heap, -dist)
            if len(max_heap) < k:
                results.append(-1)
            else:
                results.append(-max_heap[0])
        return results