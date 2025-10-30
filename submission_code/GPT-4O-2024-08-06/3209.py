from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        total_cost = 0
        i = 0
        
        while i < n:
            # Purchase the current fruit
            total_cost += prices[i]
            # Skip the next i fruits as they can be taken for free
            i += i + 1
        
        return total_cost