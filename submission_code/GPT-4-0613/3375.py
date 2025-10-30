import heapq
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        heap = coins[:]
        heapq.heapify(heap)
        count = 0
        while heap:
            val = heapq.heappop(heap)
            count += 1
            if count == k:
                return val
            for coin in coins:
                if coin > val and (coin not in heap):
                    heapq.heappush(heap, coin)