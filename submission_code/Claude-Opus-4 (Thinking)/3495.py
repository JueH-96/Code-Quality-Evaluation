class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq
        
        results = []
        max_heap = []  # Stores negative distances to simulate max heap
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            
            # If we have less than k obstacles, add this one
            if len(max_heap) < k:
                heapq.heappush(max_heap, -distance)
            # If this obstacle is closer than the k-th closest so far
            elif distance < -max_heap[0]:
                heapq.heapreplace(max_heap, -distance)
            
            # Get k-th nearest distance
            if len(max_heap) < k:
                results.append(-1)
            else:
                results.append(-max_heap[0])  # Convert back to positive
        
        return results