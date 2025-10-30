class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        total_cost = 0
        i = 0
        
        while i < n:
            total_cost += prices[i]
            i += prices[i] + 1  # Move to the next fruit that is not free
            
        return total_cost