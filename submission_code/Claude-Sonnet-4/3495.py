class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq
        
        max_heap = []  # Will store negative values to simulate max-heap
        results = []
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            
            if len(max_heap) < k:
                # Haven't reached k obstacles yet, just add
                heapq.heappush(max_heap, -distance)  # negative for max-heap
            else:
                # We have k obstacles, check if current distance is smaller than the largest
                if distance < -max_heap[0]:  # -max_heap[0] is the actual largest distance
                    heapq.heappop(max_heap)  # Remove the largest
                    heapq.heappush(max_heap, -distance)  # Add current distance
            
            if len(max_heap) < k:
                results.append(-1)
            else:
                results.append(-max_heap[0])  # The k-th smallest (largest in our max-heap)
        
        return results