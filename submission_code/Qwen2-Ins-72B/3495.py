import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        
        for x, y in queries:
            distance = abs(x) + abs(y)
            heapq.heappush(heap, distance)
            
            if len(heap) < k:
                results.append(-1)
            else:
                if len(heap) > k:
                    heapq.heappop(heap)
                results.append(heap[0])
                
        return results