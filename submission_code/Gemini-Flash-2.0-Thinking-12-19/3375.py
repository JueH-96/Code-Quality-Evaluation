import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        min_heap = []
        visited = set()
        for coin in coins:
            heapq.heappush(min_heap, (coin, coin))
        
        count = 0
        while count < k:
            current_amount, coin_value = heapq.heappop(min_heap)
            if current_amount in visited:
                continue
            visited.add(current_amount)
            count += 1
            if count == k:
                return current_amount
            heapq.heappush(min_heap, (current_amount + coin_value, coin_value))
        return -1 # Should not reach here as k is always valid