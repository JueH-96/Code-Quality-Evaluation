class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq
        
        # This will store the distances in a min-heap
        min_heap = []
        results = []
        
        for x, y in queries:
            # Calculate the Manhattan distance from the origin
            distance = abs(x) + abs(y)
            
            # Push the distance into the heap
            heapq.heappush(min_heap, -distance)  # We use negative to simulate a max-heap
            
            # If the number of obstacles is less than k, we cannot find the k-th nearest
            if len(min_heap) < k:
                results.append(-1)
            else:
                # The k-th nearest is the k-th largest, which is the root of the max-heap
                results.append(-min_heap[0])  # Convert back to positive since we stored negative distances
        
        return results