from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # max_heap will store the negatives of Manhattan distances of obstacles.
        # This way, the "smallest" element in max_heap (i.e. the one with the most negative value)
        # represents the kth smallest Manhattan distance (largest among the k smallest distances).
        max_heap = []
        results = []
        
        for query in queries:
            x, y = query
            dist = abs(x) + abs(y)
            
            # If we have fewer than k obstacles recorded, push the new obstacle.
            if len(max_heap) < k:
                heapq.heappush(max_heap, -dist)
            else:
                # If the new obstacle is closer than the kth nearest obstacle (currently at the heap top),
                # then replace the current kth with this new closer obstacle.
                if -max_heap[0] > dist:
                    heapq.heapreplace(max_heap, -dist)
            
            # If we don't have k obstacles yet, append -1.
            if len(max_heap) < k:
                results.append(-1)
            else:
                # The kth smallest obstacle's Manhattan distance is the negative of the root of the max_heap.
                results.append(-max_heap[0])
                
        return results