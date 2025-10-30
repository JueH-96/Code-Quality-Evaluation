from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        distances = []
        results = []
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            heapq.heappush(distances, distance)
            
            if len(distances) < k:
                results.append(-1)
            else:
                # Get the k-th smallest distance
                kth_distance = heapq.nsmallest(k, distances)[-1]
                results.append(kth_distance)
        
        return results