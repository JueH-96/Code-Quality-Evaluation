from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        def manhattan_distance(x, y):
            return abs(x) + abs(y)

        results = []
        obstacles = []

        for x, y in queries:
            distance = manhattan_distance(x, y)
            heapq.heappush(obstacles, distance)

            if len(obstacles) < k:
                results.append(-1)
            else:
                results.append(obstacles[k-1])

        return results