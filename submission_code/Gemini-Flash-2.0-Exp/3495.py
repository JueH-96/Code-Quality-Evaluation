class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []
        for query in queries:
            x, y = query
            obstacles.append((x, y))
            distances = []
            for obs_x, obs_y in obstacles:
                distances.append(abs(obs_x) + abs(obs_y))
            distances.sort()
            if len(distances) < k:
                results.append(-1)
            else:
                results.append(distances[k - 1])
        return results