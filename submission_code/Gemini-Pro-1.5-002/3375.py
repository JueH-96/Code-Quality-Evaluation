class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        amounts = set()
        for coin in coins:
            for i in range(1, k + 1):
                amounts.add(coin * i)
        
        sorted_amounts = sorted(list(amounts))
        
        if k <= len(sorted_amounts):
            return sorted_amounts[k-1]
        else:
            heap = []
            for coin in coins:
                heapq.heappush(heap, coin)

            count = 0
            last_val = 0
            while count < k:
                val = heapq.heappop(heap)
                if val != last_val:
                    count += 1
                    last_val = val
                
                for coin in coins:
                    heapq.heappush(heap, val + coin)
            
            return last_val