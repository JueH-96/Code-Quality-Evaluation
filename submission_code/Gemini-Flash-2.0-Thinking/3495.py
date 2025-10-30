class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        results = []
        obstacles_distances = []
        for query in queries:
            x, y = query
            distance = abs(x) + abs(y)
            obstacles_distances.append(distance)
            obstacles_distances.sort()
            if len(obstacles_distances) < k:
                results.append(-1)
            else:
                results.append(obstacles_distances[k - 1])
        return results