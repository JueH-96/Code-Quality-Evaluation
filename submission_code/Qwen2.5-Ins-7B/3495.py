from heapq import heappush, heappop

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []
        for x, y in queries:
            distance = abs(x) + abs(y)
            heappush(obstacles, (distance, x, y))
            if len(obstacles) > k:
                heappop(obstacles)
            results.append(obstacles[0][0] if len(obstacles) == k else -1)
        return results