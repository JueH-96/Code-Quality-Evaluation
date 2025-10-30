from typing import List
import heapq

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        a.sort()
        heap = []
        for i in range(len(b)):
            if len(heap) < 4:
                heapq.heappush(heap, b[i])
            else:
                if b[i] > heap[0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, b[i])
        return sum(a[i]*heap[i] for i in range(4))