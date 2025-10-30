from typing import List
from functools import lru_cache

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        
        @lru_cache(maxsize=None)
        def dfs(i):
            if i >= n:
                return 0
            min_cost = float('inf')
            for j in range(i, n):
                # The next position to consider after taking j's offer
                next_pos = 2 * j + 2
                current_cost = prices[j] + dfs(next_pos)
                if current_cost < min_cost:
                    min_cost = current_cost
            return min_cost
        
        return dfs(0)