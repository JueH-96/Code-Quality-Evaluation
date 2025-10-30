import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Remove duplicates and sort the coins
        unique_coins = sorted(list(set(coins)))
        n = len(unique_coins)
        
        # Min-heap to store the current smallest multiples
        heap = []
        for coin in unique_coins:
            heapq.heappush(heap, (coin, coin, 0))  # (current_value, coin, index)
        
        result = 0
        for _ in range(k):
            current_value, coin, index = heapq.heappop(heap)
            result = current_value
            # Push the next multiple of the same coin
            next_value = current_value + coin
            heapq.heappush(heap, (next_value, coin, index + 1))
        
        return result