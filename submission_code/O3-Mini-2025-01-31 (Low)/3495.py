from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # We will use a max heap (implemented using negatives) to maintain the smallest k Manhattan distances.
        max_heap = []
        results = []
        
        for x, y in queries:
            d = abs(x) + abs(y)
            if len(max_heap) < k:
                # We haven't reached k obstacles yet.
                heapq.heappush(max_heap, -d)
            else:
                # If the new distance is smaller than the kth smallest (which is -max_heap[0]),
                # then replace it with the new smaller distance.
                if -max_heap[0] > d:
                    heapq.heapreplace(max_heap, -d)
            # After each query, if we have less than k obstacles, answer is -1.
            if len(max_heap) < k:
                results.append(-1)
            else:
                # kth smallest obstacle's distance is the largest value in the heap.
                results.append(-max_heap[0] * -1)
                
        return results

# Example usage:
# sol = Solution()
# print(sol.resultsArray([[1,2],[3,4],[2,3],[-3,0]], 2))  # Output: [-1,7,5,3]
# print(sol.resultsArray([[5,5],[4,4],[3,3]], 1))         # Output: [10,8,6]