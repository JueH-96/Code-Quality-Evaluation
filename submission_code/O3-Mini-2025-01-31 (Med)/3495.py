from typing import List
import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # We use a max-heap (simulated using negative values in a min-heap)
        # to maintain the k smallest Manhattan distances.
        max_heap = []
        results = []
        
        for x, y in queries:
            # Calculate Manhattan distance from the origin.
            d = abs(x) + abs(y)
            
            # If we haven't added k obstacles yet, simply add the new obstacle.
            if len(max_heap) < k:
                heapq.heappush(max_heap, -d)
            else:
                # When there are already k obstacles in our heap,
                # the kth closest is the maximum in the heap (i.e., -max_heap[0]).
                # If the new distance d is smaller than this kth nearest,
                # it should be included among the k closest distances.
                if d < -max_heap[0]:
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, -d)
            
            # If we have less than k obstacles, output -1.
            # Otherwise, the kth nearest obstacle's distance is -max_heap[0].
            if len(max_heap) < k:
                results.append(-1)
            else:
                results.append(-max_heap[0])
                
        return results

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    queries1 = [[1,2],[3,4],[2,3],[-3,0]]
    k1 = 2
    print(sol.resultsArray(queries1, k1))  # Expected output: [-1, 7, 5, 3]

    # Example 2:
    queries2 = [[5,5],[4,4],[3,3]]
    k2 = 1
    print(sol.resultsArray(queries2, k2))  # Expected output: [10, 8, 6]