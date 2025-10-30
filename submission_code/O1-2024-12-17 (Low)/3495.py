import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # We maintain a max-heap of size up to k using negative values
        # so that the "top" is the k-th smallest distance (the largest in this max subset).
        max_heap = []
        results = []
        
        for x, y in queries:
            dist = abs(x) + abs(y)
            
            # If we have fewer than k elements, just push
            if len(max_heap) < k:
                heapq.heappush(max_heap, -dist)
            else:
                # If the new distance is smaller than the largest in top k,
                # pop and push the new distance
                if dist < -max_heap[0]:
                    heapq.heapreplace(max_heap, -dist)
            
            # If we still have fewer than k elements, the answer is -1
            if len(max_heap) < k:
                results.append(-1)
            else:
                # The k-th nearest obstacle is the largest in our top k
                results.append(-max_heap[0])
        
        return results