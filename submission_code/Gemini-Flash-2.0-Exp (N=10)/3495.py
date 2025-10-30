import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []
        for x, y in queries:
            dist = abs(x) + abs(y)
            heapq.heappush(obstacles, dist)
            if len(obstacles) < k:
                results.append(-1)
            else:
                kth_nearest = heapq.nlargest(k, obstacles)[-1]
                results.append(kth_nearest)
        return results