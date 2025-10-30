import bisect

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []
        distances = []
        for query in queries:
            x, y = query
            obstacles.append(query)
            distance = abs(x) + abs(y)
            bisect.insort(distances, distance)
            if len(distances) < k:
                results.append(-1)
            else:
                results.append(distances[k-1])
        return results