from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        k_heap = []  # max-heap implemented as min-heap with negated values
        others_heap = []  # min-heap for distances larger than the k-th smallest
        results = []
        
        for x, y in queries:
            d = abs(x) + abs(y)
            if len(k_heap) < k:
                heapq.heappush(k_heap, -d)
            else:
                current_max = -k_heap[0]
                if d < current_max:
                    # Replace the current_max with d in k_heap
                    popped = -heapq.heappop(k_heap)
                    heapq.heappush(k_heap, -d)
                    heapq.heappush(others_heap, popped)
                else:
                    heapq.heappush(others_heap, d)
            
            total_obstacles = len(k_heap) + len(others_heap)
            if total_obstacles < k:
                results.append(-1)
            else:
                results.append(-k_heap[0])
        
        return results