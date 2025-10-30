class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        import heapq

        heap = []  # Max heap using negative distances
        results = []

        for x, y in queries:
            D = abs(x) + abs(y)
            if len(heap) < k:
                heapq.heappush(heap, -D)
            else:
                if D < -heap[0]:
                    heapq.heappushpop(heap, -D)
            if len(heap) < k:
                results.append(-1)
            else:
                results.append(-heap[0])
        return results