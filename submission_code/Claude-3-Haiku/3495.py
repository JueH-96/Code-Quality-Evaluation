from typing import List
from heapq import heappush, heappop

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []

        for x, y in queries:
            # Calculate the distance of the new obstacle from the origin
            distance = abs(x) + abs(y)

            # Add the new obstacle to the heap
            heappush(obstacles, (distance, x, y))

            # Find the k-th nearest obstacle
            k_nearest = []
            for _ in range(k):
                if obstacles:
                    k_nearest.append(heappop(obstacles))
                else:
                    results.append(-1)
                    break
            else:
                results.append(k_nearest[-1][0])

            # Put the k-th nearest obstacles back into the heap
            for d, x, y in k_nearest:
                heappush(obstacles, (d, x, y))

        return results