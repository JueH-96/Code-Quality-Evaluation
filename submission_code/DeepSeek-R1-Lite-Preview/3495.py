from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = set()
        heap = []
        results = []
        for query in queries:
            x, y = query
            coord = (x, y)
            if coord not in obstacles:
                obstacles.add(coord)
                distance = abs(x) + abs(y)
                if len(heap) < k:
                    heapq.heappush(heap, -distance)
                else:
                    if distance < -heap[0]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, -distance)
                if len(heap) >= k:
                    kth_distance = -heap[0]
                    results.append(kth_distance)
                else:
                    results.append(-1)
        return results