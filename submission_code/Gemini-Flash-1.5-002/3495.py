class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []
        for x, y in queries:
            obstacles.append((abs(x) + abs(y), (x,y)))
            obstacles.sort()
            if len(obstacles) < k:
                results.append(-1)
            else:
                results.append(obstacles[k-1][0])
        return results