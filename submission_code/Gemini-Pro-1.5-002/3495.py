class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        distances = []
        results = []
        for x, y in queries:
            distance = abs(x) + abs(y)
            distances.append(distance)
            distances.sort()
            if len(distances) < k:
                results.append(-1)
            else:
                results.append(distances[k - 1])
        return results