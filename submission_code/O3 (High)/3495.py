from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # max-heap (implemented with negated values) that keeps the k   nearest distances
        small_max_heap: List[int] = []     # contains the k smallest distances (negated)
        # min-heap that keeps all larger distances
        large_min_heap: List[int] = []     # contains the remaining distances
        
        results: List[int] = []
        
        for x, y in queries:
            dist = abs(x) + abs(y)
            
            if len(small_max_heap) < k:
                # We still don’t have k elements – just push into the “small” heap.
                heapq.heappush(small_max_heap, -dist)
            else:
                # We already have k candidates – decide where the new distance belongs.
                if dist < -small_max_heap[0]:
                    # Goes into the k-smallest set.
                    heapq.heappush(small_max_heap, -dist)
                    # Remove the (k+1)-th smallest and move it to the “large” heap.
                    moved = -heapq.heappop(small_max_heap)
                    heapq.heappush(large_min_heap, moved)
                else:
                    # New distance is larger than the k-th smallest.
                    heapq.heappush(large_min_heap, dist)
            
            # Append the current answer.
            if len(small_max_heap) < k:
                results.append(-1)
            else:
                results.append(-small_max_heap[0])   # k-th nearest distance
        
        return results