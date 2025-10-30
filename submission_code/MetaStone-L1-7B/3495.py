import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        for query in queries:
            x, y = query
            d = abs(x) + abs(y)
            if len(heap) < k:
                heapq.heappush(heap, -d)
            else:
                current_max = -heap[0]
                if d < current_max:
                    popped = heapq.heappop(heap)
                    heapq.heappush(heap, -d)
            if len(heap) >= k:
                results.append(-heap[0])
            else:
                results.append(-1)
        return results