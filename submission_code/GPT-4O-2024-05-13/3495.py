from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        results = []
        min_heap = []
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            heapq.heappush(min_heap, distance)
            
            if len(min_heap) < k:
                results.append(-1)
            else:
                # Get the k-th smallest element in the heap
                if len(min_heap) > k:
                    heapq.heappop(min_heap)
                results.append(min_heap[0])
        
        return results