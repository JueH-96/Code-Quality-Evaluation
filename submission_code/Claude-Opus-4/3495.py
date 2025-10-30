class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq
        
        # Max heap to store the k nearest obstacles
        # Python has min heap by default, so we'll use negative distances
        max_heap = []
        results = []
        
        for x, y in queries:
            # Calculate Manhattan distance from origin
            distance = abs(x) + abs(y)
            
            # Add to heap (negative for max heap behavior)
            heapq.heappush(max_heap, -distance)
            
            # If we have more than k obstacles, remove the farthest
            if len(max_heap) > k:
                heapq.heappop(max_heap)
            
            # Check if we have k obstacles
            if len(max_heap) < k:
                results.append(-1)
            else:
                # The k-th nearest is the maximum in our heap of k elements
                # (which is the root of the max heap)
                results.append(-max_heap[0])
        
        return results