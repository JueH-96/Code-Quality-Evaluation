from typing import List
import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        min_heap = []
        for coin in coins:
            heapq.heappush(min_heap, (coin, coin))
        
        for _ in range(k - 1):
            val, coin = heapq.heappop(min_heap)
            heapq.heappush(min_heap, (val + coin, coin))
        
        return min_heap[0][0]