import bisect

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        distances = []
        results = []
        for query in queries:
            x, y = query
            distance = abs(x) + abs(y)
            bisect.insort(distances, distance)
            if len(distances) >= k:
                results.append(distances[k-1])
            else:
                results.append(-1)
        return results