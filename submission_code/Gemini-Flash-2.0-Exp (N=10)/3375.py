import heapq
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        heap = []
        seen = set()
        for coin in coins:
            heapq.heappush(heap, (coin, coin))
            seen.add(coin)
        
        count = 0
        result = 0
        while count < k:
            result, coin = heapq.heappop(heap)
            count += 1
            next_val = result + coin
            if next_val not in seen:
                heapq.heappush(heap, (next_val, coin))
                seen.add(next_val)
        
        return result