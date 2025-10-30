from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        distances = []
        results = []
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            heapq.heappush(distances, distance)
            
            if len(distances) >= k:
                kth_distance = heapq.nsmallest(k, distances)[-1]
                results.append(kth_distance)
            else:
                results.append(-1)
        
        return results