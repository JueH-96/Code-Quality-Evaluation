from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []
        for x, y in queries:
            dist = abs(x) + abs(y)
            heapq.heappush(obstacles, -dist)
            if len(obstacles) > k:
                heapq.heappop(obstacles)
            results.append(-obstacles[0] if len(obstacles) == k else -1)
        return results