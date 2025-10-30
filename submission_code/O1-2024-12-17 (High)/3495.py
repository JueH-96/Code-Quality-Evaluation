from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # We'll keep a max-heap of size <= k to track the k smallest distances.
        # In Python, we can simulate a max-heap by pushing negative values onto a min-heap.
        
        results = []
        max_heap = []  # will store negative values to simulate a max-heap
        
        for x, y in queries:
            dist = abs(x) + abs(y)
            if len(max_heap) < k:
                # Push onto the heap
                heapq.heappush(max_heap, -dist)
                
                # If we don't yet have k elements, the k-th nearest distance doesn't exist
                if len(max_heap) < k:
                    results.append(-1)
                else:
                    # Now we have k elements, the top of this max-heap is the k-th smallest distance
                    results.append(-max_heap[0])
            else:
                # If the new distance is smaller than the largest among the k smallest,
                # replace it so the heap always has the k smallest distances
                if dist < -max_heap[0]:
                    heapq.heapreplace(max_heap, -dist)
                    
                # The k-th smallest distance is at the root of the max-heap
                results.append(-max_heap[0])
        
        return results