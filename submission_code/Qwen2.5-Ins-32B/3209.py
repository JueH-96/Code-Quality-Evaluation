from typing import List
from functools import lru_cache

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        @lru_cache(maxsize=None)
        def dp(index):
            if index >= len(prices):
                return 0
            min_cost = float('inf')
            for i in range(index, min(index + index + 1, len(prices))):
                min_cost = min(min_cost, prices[index] + dp(i + 1))
            return min_cost
        
        return dp(0)