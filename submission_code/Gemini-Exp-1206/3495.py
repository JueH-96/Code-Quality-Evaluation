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
                temp = []
                for _ in range(k):
                    temp.append(heapq.heappop(obstacles))
                results.append(temp[-1])
                for dist in temp:
                    heapq.heappush(obstacles, dist)
        return results