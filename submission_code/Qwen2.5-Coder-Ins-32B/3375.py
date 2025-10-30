from typing import List
import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        min_heap = []
        seen = set()
        
        for coin in coins:
            heapq.heappush(min_heap, coin)
            seen.add(coin)
        
        current = 0
        for _ in range(k):
            current = heapq.heappop(min_heap)
            for coin in coins:
                new_val = current + coin
                if new_val not in seen:
                    seen.add(new_val)
                    heapq.heappush(min_heap, new_val)
        
        return current