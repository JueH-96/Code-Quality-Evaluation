import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        obstacles = []
        results = []
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            heapq.heappush(obstacles, -distance)  # Use a max-heap to keep track of the k closest obstacles
            
            if len(obstacles) > k:
                heapq.heappop(obstacles)  # Remove the farthest obstacle if we have more than k
            
            if len(obstacles) < k:
                results.append(-1)  # Not enough obstacles
            else:
                results.append(-obstacles[0])  # The k-th closest obstacle distance
        
        return results