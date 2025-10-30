from typing import List
import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        min_heap = []
        for coin in coins:
            heapq.heappush(min_heap, coin)

        current_amount = 0
        for _ in range(k):
            current_amount = heapq.heappop(min_heap)
            for coin in coins:
                new_amount = current_amount + coin
                heapq.heappush(min_heap, new_amount)
                if new_amount >= current_amount + coin:  # Avoid duplicates
                    break

        return current_amount