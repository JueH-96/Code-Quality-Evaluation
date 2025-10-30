import heapq
from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        heap = []
        res = 0
        for price in prices:
            heapq.heappush(heap, -price)
            if len(heap) > 1:
                res -= heapq.heappop(heap)
        return res