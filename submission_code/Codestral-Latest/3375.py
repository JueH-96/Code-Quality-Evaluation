class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        from heapq import heappop, heappush

        # Initialize a min-heap with the first multiples of each coin
        heap = [(coin, coin) for coin in coins]
        heappush(heap, (0, 0))  # Push a dummy value to handle the case when k is 1

        # Initialize a set to keep track of seen amounts
        seen = set()

        # Pop the smallest element from the heap k times
        for _ in range(k):
            amount, coin = heappop(heap)
            if amount not in seen:
                seen.add(amount)
                # Push the next multiple of the same coin
                heappush(heap, (amount + coin, coin))

        return amount