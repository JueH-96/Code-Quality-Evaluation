import heapq

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        max_heap = []
        results = []
        for x, y in queries:
            distance = abs(x) + abs(y)
            if len(max_heap) < k:
                heapq.heappush(max_heap, -distance)
            else:
                if distance < -max_heap[0]:
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, -distance)
            if len(max_heap) >= k:
                results.append(-max_heap[0])
            else:
                results.append(-1)
        return results