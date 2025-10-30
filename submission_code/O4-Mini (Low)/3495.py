import heapq
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # max_heap will store the k smallest distances as negatives (so the largest of them is at the top)
        max_heap = []
        # min_heap will store the rest of the distances
        min_heap = []
        results = []
        
        for x, y in queries:
            d = abs(x) + abs(y)
            
            # If we haven't yet collected k distances, just push onto max_heap
            if len(max_heap) < k:
                heapq.heappush(max_heap, -d)
            else:
                # We already have k in max_heap; check if this new distance
                # should be in the k smallest
                if d < -max_heap[0]:
                    # Push it onto max_heap, then move the largest from max_heap to min_heap
                    heapq.heappush(max_heap, -d)
                    largest_of_k = -heapq.heappop(max_heap)
                    heapq.heappush(min_heap, largest_of_k)
                else:
                    # This distance is not in the k smallest, goes to min_heap
                    heapq.heappush(min_heap, d)
            
            # The answer after this query is the k-th smallest iff we have at least k obstacles
            if len(max_heap) < k:
                results.append(-1)
            else:
                # The k-th smallest is the largest in our max_heap
                results.append(-max_heap[0])
        
        return results

# Example usage:
# sol = Solution()
# print(sol.resultsArray([[1,2],[3,4],[2,3],[-3,0]], 2))  # [-1,7,5,3]
# print(sol.resultsArray([[5,5],[4,4],[3,3]], 1))         # [10,8,6]