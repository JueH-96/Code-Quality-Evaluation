import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # We'll maintain a max-heap of size at most k using Python's min-heap
        # by storing negative distances. The largest distance in the k smallest
        # is always at the top (heap[0]) as the smallest negative number.
        
        max_heap = []
        results = []
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            
            if len(max_heap) < k:
                # Push the negative distance
                heapq.heappush(max_heap, -distance)
            else:
                # Only replace if this new distance is smaller
                # than the current largest distance in the k smallest
                if distance < -max_heap[0]:
                    heapq.heapreplace(max_heap, -distance)
            
            # If we don't have k obstacles yet, answer is -1
            if len(max_heap) < k:
                results.append(-1)
            else:
                # k-th nearest is the largest among the k smallest => -max_heap[0]
                results.append(-max_heap[0])
        
        return results