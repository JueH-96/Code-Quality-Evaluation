import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        
        results = []
        # This will be a min-heap of negative distances, effectively simulating a max-heap of positive distances.
        # It will store the k smallest distances encountered so far.
        k_smallest_heap = []

        for x, y in queries:
            distance = abs(x) + abs(y)

            if len(k_smallest_heap) < k:
                # If the heap is not yet full, add the new distance.
                heapq.heappush(k_smallest_heap, -distance)
            elif distance < -k_smallest_heap[0]:
                # If the heap is full and the new distance is smaller than the largest
                # element in the heap (the k-th smallest distance), replace the largest
                # element with the new one. `heapreplace` is an efficient pop-then-push.
                heapq.heapreplace(k_smallest_heap, -distance)
            
            # After each query, determine the result for the current state.
            if len(k_smallest_heap) == k:
                # If the heap has k elements, the k-th smallest distance is the largest
                # element in the heap. In our min-heap of negative values, this is -heap[0].
                results.append(-k_smallest_heap[0])
            else:
                # If there are fewer than k obstacles, the answer is -1.
                results.append(-1)
                
        return results