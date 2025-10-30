from heapq import heappush, heappushpop
from typing import List

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        results = []
        count = 0
        for x, y in queries:
            d = abs(x) + abs(y)
            if count < k:
                heappush(heap, -d)
                count += 1
                if count < k:
                    results.append(-1)
                else:
                    results.append(-heap[0])
            else:
                if d < -heap[0]:
                    heappushpop(heap, -d)
                results.append(-heap[0])
        return results