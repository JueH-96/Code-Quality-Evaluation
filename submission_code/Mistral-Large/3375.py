from typing import List
import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Priority queue to store (amount, coin_index)
        pq = []

        # Initialize the priority queue with the smallest multiple of each coin
        for i, coin in enumerate(coins):
            heapq.heappush(pq, (coin, i))

        # Variable to keep track of the k-th smallest amount
        kth_smallest = 0

        # Extract the smallest amount k times
        for _ in range(k):
            amount, coin_index = heapq.heappop(pq)
            kth_smallest = amount
            # Push the next multiple of the same coin back into the queue
            next_amount = amount + coins[coin_index]
            heapq.heappush(pq, (next_amount, coin_index))

        return kth_smallest