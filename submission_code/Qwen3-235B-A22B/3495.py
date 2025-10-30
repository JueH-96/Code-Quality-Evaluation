from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        for x, y in queries:
            distance = abs(x) + abs(y)
            if len(heap) < k:
                heapq.heappush(heap, -distance)
            else:
                current_largest = -heap[0]
                if distance < current_largest:
                    heapq.heappop(heap)
                    heapq.heappush(heap, -distance)
            # Append the result based on the current heap size
            if len(heap) < k:
                results.append(-1)
            else:
                results.append(-heap[0])
        return results