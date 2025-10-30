import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        min_heap = []
        seen = set()

        for coin in coins:
            if coin not in seen:
                heapq.heappush(min_heap, coin)
                seen.add(coin)

        count = 0
        while count < k:
            smallest = heapq.heappop(min_heap)
            count += 1

            if count == k:
                return smallest

            for coin in coins:
                next_multiple = smallest + coin
                if next_multiple not in seen:
                    heapq.heappush(min_heap, next_multiple)
                    seen.add(next_multiple)

        return -1 # Should not reach here