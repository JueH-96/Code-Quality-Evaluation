from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []

        for x, y in queries:
            distance = abs(x) + abs(y)
            heapq.heappush(obstacles, distance)

            if len(obstacles) < k:
                results.append(-1)
            else:
                while len(obstacles) > k:
                    heapq.heappop(obstacles)
                results.append(obstacles[0])

        return results