from sortedcontainers import SortedList

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = SortedList()
        results = []

        for x, y in queries:
            distance = abs(x) + abs(y)
            obstacles.add(distance)

            if len(obstacles) < k:
                results.append(-1)
            else:
                results.append(obstacles[k-1])

        return results