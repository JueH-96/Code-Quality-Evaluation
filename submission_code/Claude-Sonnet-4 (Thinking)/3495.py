class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq
        
        max_heap = []
        results = []
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            
            if len(max_heap) < k:
                heapq.heappush(max_heap, -distance)
            elif -max_heap[0] > distance:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, -distance)
            
            if len(max_heap) < k:
                results.append(-1)
            else:
                results.append(-max_heap[0])
        
        return results