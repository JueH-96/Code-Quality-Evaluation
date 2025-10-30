class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        import heapq
        heap = []
        for coin in coins:
            heapq.heappush(heap, coin)
        
        amounts = set()
        
        
        q = [(0,tuple())]
        visited = set()
        
        for _ in range(k):
            curr_sum, curr_coins = heapq.heappop(q)
            
            amounts.add(curr_sum)
            
            for coin in coins:
                new_sum = curr_sum + coin
                new_coins = tuple(sorted(list(curr_coins) + [coin]))
                if (new_sum, new_coins) not in visited:
                    visited.add((new_sum, new_coins))
                    heapq.heappush(q,(new_sum, new_coins))
                    
        return list(sorted(list(amounts)))[k-1]