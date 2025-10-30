class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        res = []
        obstacles = set()
        for query in queries:
            x, y = query
            obstacles.add((abs(x), abs(y)))
            if len(obstacles) < k:
                res.append(-1)
            else:
                min_heap = []
                for obs in obstacles:
                    dist = obs[0] + obs[1]
                    heapq.heappush(min_heap, (dist, obs))
                    if len(min_heap) > k:
                        heapq.heappop(min_heap)
                res.append(min_heap[0][0])
        return res